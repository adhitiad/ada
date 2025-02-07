from typing import Optional

from redis import Redis


def create_redis_connection(
    host="redis-12486.c334.asia-southeast2-1.gce.redns.redis-cloud.com",
    port=12486,
    decode_responses=True,
    username="default",
    password="UHW0bZalOZCazpkn8kGPtkmMJC4LUP9z",
    db: int = 0,
) -> Redis:
    """
    Create a Redis connection.

    Args:
        host (str): The Redis server host. Defaults to "localhost".
        port (int): The Redis server port. Defaults to 6379.
        db (int): The Redis database number. Defaults to 0.

    Returns:
        redis.Redis: The Redis connection object.
    """
    try:
        redis_client = Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=decode_responses,
            username=username,
            password=password,
        )
        redis_client.ping()  # Test the connection
        print("Connected to Redis", redis_client.info())
        return redis_client
    except ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def close_redis_connection(redis_client: Redis):
    """
    Close the Redis connection.

    Args:
        redis_client (redis.Redis): The Redis connection object.
    """
    try:
        redis_client.close()
    except Exception as e:
        print(f"Error closing Redis connection: {e}")
