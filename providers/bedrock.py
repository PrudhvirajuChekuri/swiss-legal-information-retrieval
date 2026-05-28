"""AWS Bedrock client factory with retry logic."""

import json
import time

import boto3

from config import AWS_REGION


def get_client() -> boto3.client:
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def converse(
    client,
    model_id: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    prefill: str | None = None,
    retries: int = 3,
    delay: float = 5.0,
) -> str:
    """Call Bedrock converse API with optional assistant prefill and retry."""
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    if prefill:
        messages.append({"role": "assistant", "content": [{"text": prefill}]})

    for attempt in range(1, retries + 1):
        try:
            response = client.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=messages,
                inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            )
            return response["output"]["message"]["content"][0]["text"].strip()
        except Exception as e:
            print(f"converse attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
    return ""


def invoke(
    client,
    model_id: str,
    body: dict,
    retries: int = 3,
    delay: float = 5.0,
) -> dict:
    """Call Bedrock invoke_model API with retry."""
    for attempt in range(1, retries + 1):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())
        except Exception as e:
            print(f"invoke attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
    return {}
