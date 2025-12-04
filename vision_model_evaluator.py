"""
Vision Model Evaluator for Traffic Sign Recognition
====================================================

A production-ready toolkit for evaluating vision-language models on image classification tasks.
Supports: Google Gemini (replaces deprecated Bard), OpenAI GPT-4V, and local LLaVA models.

Features:
- Async processing for improved throughput
- Automatic retry with exponential backoff
- Progress tracking with tqdm
- Comprehensive logging
- Configurable via YAML or environment variables
- Results export to CSV/JSON with metadata
- Batch processing with rate limiting

Usage:
    python vision_model_evaluator.py --config config.yaml --model gpt4v --input ./images
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ModelType(str, Enum):
    """Supported model types."""
    GEMINI = "gemini"
    GPT4V = "gpt4v"
    LLAVA = "llava"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_type: ModelType
    api_key: Optional[str] = None
    model_name: str = ""
    base_url: str = ""
    max_tokens: int = 100
    temperature: float = 0.1
    timeout: float = 60.0
    
    def __post_init__(self):
        """Set defaults based on model type."""
        if self.model_type == ModelType.GEMINI:
            self.model_name = self.model_name or "gemini-1.5-flash"
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        elif self.model_type == ModelType.GPT4V:
            self.model_name = self.model_name or "gpt-4o"
            self.base_url = "https://api.openai.com/v1"


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""
    input_path: Path
    output_path: Path
    prompt: str = (
        "Is the traffic sign displayed a real-world traffic sign that has the same "
        "shape, color, pattern and text as a real world traffic sign? "
        "Answer with 'yes' or 'no' and provide a brief explanation."
    )
    supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 1.0
    batch_size: int = 10
    save_intermediate: bool = True


@dataclass
class EvaluationResult:
    """Result from evaluating a single image."""
    image_name: str
    image_path: str
    prediction: str
    raw_response: str
    model_type: str
    model_name: str
    timestamp: str
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Utility Functions
# =============================================================================

def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: Path) -> str:
    """Get MIME type based on file extension."""
    extension_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return extension_map.get(image_path.suffix.lower(), "image/jpeg")


def discover_images(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Discover all image files in a folder."""
    images = []
    for ext in extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


# =============================================================================
# Base Model Client
# =============================================================================

class VisionModelClient(ABC):
    """Abstract base class for vision model clients."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> VisionModelClient:
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        return self
    
    async def __aexit__(self, *args) -> None:
        if self.client:
            await self.client.aclose()
    
    @abstractmethod
    async def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze an image and return the model's response."""
        pass
    
    def get_model_info(self) -> dict[str, str]:
        """Return model identification info."""
        return {
            "model_type": self.config.model_type.value,
            "model_name": self.config.model_name,
        }


# =============================================================================
# Google Gemini Client (Replaces deprecated Bard)
# =============================================================================

class GeminiClient(VisionModelClient):
    """Client for Google Gemini API (replaces deprecated Bard)."""
    
    async def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze an image using Gemini."""
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        image_data = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        url = (
            f"{self.config.base_url}/models/{self.config.model_name}:generateContent"
            f"?key={self.config.api_key}"
        )
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text from response
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            logger.warning(f"Unexpected Gemini response structure: {data}")
            raise ValueError(f"Failed to parse Gemini response: {e}")


# =============================================================================
# OpenAI GPT-4V Client
# =============================================================================

class GPT4VClient(VisionModelClient):
    """Client for OpenAI GPT-4 Vision API."""
    
    async def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze an image using GPT-4V."""
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        image_data = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        url = f"{self.config.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.warning(f"Unexpected GPT-4V response structure: {data}")
            raise ValueError(f"Failed to parse GPT-4V response: {e}")


# =============================================================================
# LLaVA Local Client
# =============================================================================

class LLaVAClient(VisionModelClient):
    """Client for local LLaVA model via llama.cpp server."""
    
    def __init__(self, config: ModelConfig, server_url: str = "http://localhost:8080"):
        super().__init__(config)
        self.server_url = server_url
    
    async def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze an image using local LLaVA."""
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        image_data = encode_image_to_base64(image_path)
        
        # Format for llama.cpp server with multimodal support
        payload = {
            "prompt": f"USER: [img-1] {prompt}\nASSISTANT:",
            "image_data": [{"data": image_data, "id": 1}],
            "n_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stop": ["USER:", "</s>"],
        }
        
        response = await self.client.post(
            f"{self.server_url}/completion",
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        return data.get("content", "").strip()


# =============================================================================
# Evaluator Engine
# =============================================================================

class VisionModelEvaluator:
    """Main evaluator class for running vision model evaluations."""
    
    def __init__(
        self,
        model_client: VisionModelClient,
        eval_config: EvaluationConfig,
    ):
        self.model_client = model_client
        self.eval_config = eval_config
        self.results: list[EvaluationResult] = []
    
    async def evaluate_single(self, image_path: Path) -> EvaluationResult:
        """Evaluate a single image with retry logic."""
        model_info = self.model_client.get_model_info()
        start_time = time.perf_counter()
        
        for attempt in range(1, self.eval_config.max_retries + 1):
            try:
                response = await self.model_client.analyze_image(
                    image_path,
                    self.eval_config.prompt
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Extract yes/no prediction
                prediction = self._extract_prediction(response)
                
                return EvaluationResult(
                    image_name=image_path.name,
                    image_path=str(image_path),
                    prediction=prediction,
                    raw_response=response,
                    model_type=model_info["model_type"],
                    model_name=model_info["model_name"],
                    timestamp=datetime.now().isoformat(),
                    latency_ms=latency_ms,
                    success=True,
                    metadata={"attempt": attempt}
                )
                
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{self.eval_config.max_retries} failed for "
                    f"{image_path.name}: {e}"
                )
                
                if attempt < self.eval_config.max_retries:
                    delay = self.eval_config.retry_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                else:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return EvaluationResult(
                        image_name=image_path.name,
                        image_path=str(image_path),
                        prediction="error",
                        raw_response="",
                        model_type=model_info["model_type"],
                        model_name=model_info["model_name"],
                        timestamp=datetime.now().isoformat(),
                        latency_ms=latency_ms,
                        success=False,
                        error_message=str(e),
                    )
        
        # Should not reach here, but satisfy type checker
        raise RuntimeError("Unexpected evaluation state")
    
    def _extract_prediction(self, response: str) -> str:
        """Extract yes/no prediction from model response."""
        response_lower = response.lower().strip()
        
        # Check for explicit yes/no at the start
        if response_lower.startswith("yes"):
            return "yes"
        elif response_lower.startswith("no"):
            return "no"
        
        # Check for yes/no anywhere in response
        yes_count = response_lower.count("yes")
        no_count = response_lower.count("no")
        
        if yes_count > no_count:
            return "yes"
        elif no_count > yes_count:
            return "no"
        
        return "unclear"
    
    async def evaluate_batch(
        self,
        images: list[Path],
        progress_bar: bool = True,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of images with rate limiting."""
        results = []
        
        iterator: Iterator[Path] = iter(images)
        if progress_bar:
            iterator = tqdm(images, desc="Evaluating images", unit="img")
        
        for image_path in iterator:
            result = await self.evaluate_single(image_path)
            results.append(result)
            self.results.append(result)
            
            # Rate limiting
            await asyncio.sleep(self.eval_config.rate_limit_delay)
            
            # Save intermediate results
            if self.eval_config.save_intermediate and len(results) % 10 == 0:
                self._save_intermediate_results()
        
        return results
    
    def _save_intermediate_results(self) -> None:
        """Save intermediate results to prevent data loss."""
        intermediate_path = self.eval_config.output_path.parent / "intermediate_results.json"
        with open(intermediate_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        logger.debug(f"Saved intermediate results to {intermediate_path}")
    
    def save_results(self, format: str = "both") -> dict[str, Path]:
        """Save final results to CSV and/or JSON."""
        saved_files = {}
        
        if format in ("csv", "both"):
            csv_path = self.eval_config.output_path.with_suffix(".csv")
            self._save_csv(csv_path)
            saved_files["csv"] = csv_path
            logger.info(f"Saved CSV results to {csv_path}")
        
        if format in ("json", "both"):
            json_path = self.eval_config.output_path.with_suffix(".json")
            self._save_json(json_path)
            saved_files["json"] = json_path
            logger.info(f"Saved JSON results to {json_path}")
        
        return saved_files
    
    def _save_csv(self, path: Path) -> None:
        """Save results to CSV format."""
        if not self.results:
            return
        
        fieldnames = [
            "image_name", "prediction", "raw_response", "model_type",
            "model_name", "timestamp", "latency_ms", "success", "error_message"
        ]
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = {k: getattr(result, k) for k in fieldnames}
                writer.writerow(row)
    
    def _save_json(self, path: Path) -> None:
        """Save results to JSON format with full metadata."""
        output = {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "total_images": len(self.results),
                "successful": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "prompt": self.eval_config.prompt,
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> dict[str, Any]:
        """Get evaluation summary statistics."""
        if not self.results:
            return {"error": "No results available"}
        
        successful = [r for r in self.results if r.success]
        predictions = [r.prediction for r in successful]
        latencies = [r.latency_ms for r in successful]
        
        return {
            "total_images": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "predictions": {
                "yes": predictions.count("yes"),
                "no": predictions.count("no"),
                "unclear": predictions.count("unclear"),
            },
            "latency_stats": {
                "mean_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
            }
        }


# =============================================================================
# CLI Interface
# =============================================================================

def create_client(model_type: ModelType, api_key: str, **kwargs) -> VisionModelClient:
    """Factory function to create appropriate model client."""
    config = ModelConfig(model_type=model_type, api_key=api_key, **kwargs)
    
    if model_type == ModelType.GEMINI:
        return GeminiClient(config)
    elif model_type == ModelType.GPT4V:
        return GPT4VClient(config)
    elif model_type == ModelType.LLAVA:
        return LLaVAClient(config, server_url=kwargs.get("server_url", "http://localhost:8080"))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


async def run_evaluation(
    model_type: ModelType,
    api_key: str,
    input_folder: Path,
    output_path: Path,
    prompt: Optional[str] = None,
    **kwargs
) -> dict[str, Any]:
    """Run a complete evaluation pipeline."""
    
    # Setup configuration
    eval_config = EvaluationConfig(
        input_path=input_folder,
        output_path=output_path,
    )
    if prompt:
        eval_config.prompt = prompt
    
    # Discover images
    images = discover_images(input_folder, eval_config.supported_extensions)
    if not images:
        raise ValueError(f"No images found in {input_folder}")
    
    logger.info(f"Found {len(images)} images to evaluate")
    
    # Run evaluation
    async with create_client(model_type, api_key, **kwargs) as client:
        evaluator = VisionModelEvaluator(client, eval_config)
        await evaluator.evaluate_batch(images)
        
        # Save results
        saved_files = evaluator.save_results()
        summary = evaluator.get_summary()
    
    return {
        "summary": summary,
        "saved_files": {k: str(v) for k, v in saved_files.items()}
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate vision models on traffic sign recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with GPT-4V
  python vision_model_evaluator.py --model gpt4v --api-key sk-xxx --input ./images --output ./results

  # Evaluate with Gemini
  python vision_model_evaluator.py --model gemini --api-key xxx --input ./images --output ./results

  # Evaluate with local LLaVA
  python vision_model_evaluator.py --model llava --server-url http://localhost:8080 --input ./images --output ./results
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["gemini", "gpt4v", "llava"],
        required=True,
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="API key (can also use GEMINI_API_KEY or OPENAI_API_KEY env vars)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input folder containing images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for results (without extension)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Custom prompt for evaluation"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8080",
        help="Server URL for LLaVA (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key from args or environment
    api_key = args.api_key
    model_type = ModelType(args.model)
    
    if not api_key and model_type == ModelType.GEMINI:
        api_key = os.environ.get("GEMINI_API_KEY")
    elif not api_key and model_type == ModelType.GPT4V:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key and model_type != ModelType.LLAVA:
        parser.error(f"API key required for {model_type.value}")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    try:
        result = asyncio.run(run_evaluation(
            model_type=model_type,
            api_key=api_key or "",
            input_folder=args.input,
            output_path=args.output,
            prompt=args.prompt,
            server_url=args.server_url,
        ))
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
