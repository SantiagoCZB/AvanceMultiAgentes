import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

async def send_coordinates(id, x, y):
    url = "http://localhost:8000/send_coordinates"
    params = {"id": id}
    data = {"x": x, "y": y}
    async with httpx.AsyncClient() as client:
        await client.post(url, params=params, json=data)

async def check_connection(n_tractors):
    url = "http://localhost:8000/check_connection"
    params = {"n_tractors": n_tractors}
    async with httpx.AsyncClient() as client:
        await client.get(url, params=params)

def run_async_in_thread(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()

def send_coordinates_background(id, x, y):
    executor.submit(run_async_in_thread, send_coordinates(id, x, y))

def check_connection_background(n_tractors):
    executor.submit(run_async_in_thread, check_connection(n_tractors))

def check_connection_sync(n_tractors):
    asyncio.run(check_connection(n_tractors))

def send_coordinates_sync(id, x, y):
    asyncio.run(send_coordinates(id, x, y))