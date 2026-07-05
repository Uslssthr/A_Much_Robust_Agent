"""
Worker 消费者 —— 独立进程运行
启动: python -m mq.task_consumer

职责：
  1. 从 Redis 队列循环取任务
  2. 分发给对应处理器
  3. 更新任务状态 & 结果
  4. 通过 Pub/Sub 推送进度
  5. 优雅关闭（处理完当前任务再退出）
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
import traceback

from backend.monitoring.metrics import (
    task_processed_total,
    task_duration_seconds,
    task_queue_length,
)

from backend.mq.redis_client import redis_client
from backend.mq.tasks import get_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Worker] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Worker:

    def __init__(self, worker_id: int = 0, concurrency: int = 3):
        self.worker_id = worker_id
        self.concurrency = concurrency
        self._running = True
        self._active = 0 # 当前处理中的任务数

    async def _process_one(self, task: dict):
        """处理单个任务"""
        task_id = task.get("task_id", "unknown")
        task_type = task.get("type", "unknown")
        start = time.time()

        self._active += 1
        logger.info(f"开始处理: task_id={task_id} type={task_type}")

        try:
            await redis_client.set_task_status(task_id, "running")

            handler = get_handler(task_type)
            if handler is None:
                raise ValueError(f"未知任务类型: {task_type}")

            # 执行处理器
            result = await handler(task)

            # 记录成功
            await redis_client.set_task_status(task_id, "success")
            await redis_client.set_task_result(task_id, {
                "status": "success",
                "result": result,
            })

            elapsed = time.time() - start
            task_processed_total.labels(type=task_type, status="success").inc()
            task_duration_seconds.labels(type=task_type).observe(elapsed)

            logger.info(
                f"完成: task_id={task_id} "
                f"elapsed={elapsed:.2f}s result={result}"
            )

        except Exception as e:
            elapsed = time.time() - start
            error_detail = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"处理任务失败: task_id={task_id} error={error_detail}\n"
                f"{traceback.format_exc()}"
            )
            await redis_client.set_task_status(task_id, "failed")
            await redis_client.set_task_result(task_id, {
                "status": "failed",
                "error": error_detail,
            })
            task_processed_total.labels(type=task_type, status="failed").inc()

        finally:
            self._active -= 1

    async def _worker_loop(self):
        """单个协程的消费循环"""
        while self._running:
            try:
                # 更新队列长度指标
                qlen = await redis_client.queue_length()
                task_queue_length.set(qlen)

                # 阻塞取任务（5秒超时，便于检查 _running）
                task = await redis_client.dequeue_task(timeout=5)
                if task is None:
                    continue

                await self._process_one(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Worker] 循环异常：{e}", exc_info=True)
                await asyncio.sleep(1)      # 避免异常时空转

    async def run(self):
        """启动 Worker（多协程并发消费）"""
        await redis_client.connect()
        logger.info(
            f"[Worker] 启动，worker_id={self.worker_id}"
            f"（并发数={self.concurrency}）"
        )

        # 启动多个消费协程
        loops = [
            asyncio.create_task(self._worker_loop())
            for _ in range(self.concurrency)
        ]

        await asyncio.gather(*loops, return_exceptions=True)

    async def shutdown(self):
        """优雅关闭"""
        logger.info("收到关闭信号，等待当前任务完成...")
        self._running = False

        # 等待活跃任务完成（最多30秒）
        for _ in range(30):
            if self._active == 0:
                break
            await asyncio.sleep(1)

        await redis_client.disconnect()
        logger.info("Worker 已关闭")


async def main():
    worker = Worker(worker_id=0, concurrency=3)

    # 注册信号处理（优雅关闭）
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows 不支持 add_signal_handler
            pass

    # 启动worker+等待停止信号
    worker_task = asyncio.create_task(worker.run())
    await stop_event.wait()

    await worker.shutdown()
    worker_task.cancel()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("被Ctrl+C中断")