# Vision Model Evaluator

A production-ready toolkit for evaluating vision-language models on image classification tasks. Specifically designed for traffic sign recognition, but adaptable to any image classification evaluation workflow.

## Features

- **Multiple Model Support**: Google Gemini, OpenAI GPT-4V/GPT-4o, and local LLaVA models
- **Async Processing**: Efficient concurrent image processing for improved throughput
- **Automatic Retry**: Exponential backoff retry logic for handling API rate limits and transient errors
- **Progress Tracking**: Real-time progress bars with tqdm
- **Comprehensive Logging**: Structured logging for debugging and monitoring
- **Result Export**: CSV and JSON export with detailed metadata
- **Batch Processing**: Configurable batch sizes with rate limiting
- **Intermediate Saves**: Auto-save results every 10 images to prevent data loss
- **Visualization**: Jupyter notebook with built-in result visualization and analysis

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### API Keys

You'll need an API key for the model you want to use:

- **OpenAI GPT-4V**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Google Gemini**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **LLaVA**: Requires local deployment (see Local LLaVA Setup section)

## Quick Start

### Command Line Interface

**Evaluate with GPT-4V:**
```bash
python vision_model_evaluator.py \
  --model gpt4v \
  --api-key sk-your-openai-key \
  --input ./images \
  --output ./results
```

**Evaluate with Gemini:**
```bash
python vision_model_evaluator.py \
  --model gemini \
  --api-key your-gemini-key \
  --input ./images \
  --output ./results
```

**Using Environment Variables:**
```bash
export OPENAI_API_KEY="sk-your-openai-key"
python vision_model_evaluator.py --model gpt4v --input ./images --output ./results
```

### Jupyter Notebook Interface

For interactive exploration and visualization:

```bash
jupyter notebook vision_model_evaluator_notebook.ipynb
```

The notebook includes:
- Interactive configuration
- Progress tracking
- Result visualization (distribution plots, latency analysis, success rates)
- Data analysis with pandas

## Usage

### Command Line Arguments

```
usage: vision_model_evaluator.py [-h] --model {gemini,gpt4v,llava}
                                  [--api-key API_KEY] --input INPUT
                                  --output OUTPUT [--prompt PROMPT]
                                  [--server-url SERVER_URL] [--verbose]

required arguments:
  --model, -m           Model to use: gemini, gpt4v, or llava
  --input, -i           Input folder containing images
  --output, -o          Output path for results (without extension)

optional arguments:
  --api-key, -k         API key (or set GEMINI_API_KEY/OPENAI_API_KEY env var)
  --prompt, -p          Custom evaluation prompt
  --server-url          LLaVA server URL (default: http://localhost:8080)
  --verbose, -v         Enable verbose logging
```

### Custom Prompts

Customize the evaluation prompt for your specific use case:

```bash
python vision_model_evaluator.py \
  --model gpt4v \
  --api-key sk-xxx \
  --input ./images \
  --output ./results \
  --prompt "Classify this traffic sign into one of the following categories: warning, regulatory, or informational."
```

### Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- WebP (`.webp`)
- GIF (`.gif`)

## Output

### CSV Format

Results are saved to `{output}.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `image_name` | Filename of the evaluated image |
| `prediction` | Extracted prediction (yes/no/unclear) |
| `raw_response` | Full model response |
| `model_type` | Model used (gemini/gpt4v/llava) |
| `model_name` | Specific model version |
| `timestamp` | ISO 8601 timestamp |
| `latency_ms` | Response time in milliseconds |
| `success` | Whether evaluation succeeded |
| `error_message` | Error details if failed |

### JSON Format

Results are also saved to `{output}.json` with additional metadata:

```json
{
  "metadata": {
    "evaluation_date": "2025-12-03T20:30:00",
    "total_images": 100,
    "successful": 98,
    "failed": 2,
    "prompt": "Is this a real traffic sign? Answer yes or no."
  },
  "results": [
    {
      "image_name": "stop_sign.jpg",
      "prediction": "yes",
      "raw_response": "Yes, this is a real traffic sign...",
      "latency_ms": 1234.56,
      ...
    }
  ]
}
```

### Intermediate Results

During evaluation, progress is automatically saved to `intermediate_results.json` every 10 images to prevent data loss.

## Configuration

### Default Evaluation Prompt

```
Is the traffic sign displayed a real-world traffic sign that has the same
shape, color, pattern and text as a real world traffic sign?
Answer with 'yes' or 'no' and provide a brief explanation.
```

### Model-Specific Settings

**Gemini:**
- Model: `gemini-1.5-flash`
- Temperature: 0.1
- Max tokens: 100

**GPT-4V:**
- Model: `gpt-4o`
- Temperature: 0.1
- Max tokens: 100
- Image detail: high

**LLaVA:**
- Requires local llama.cpp server
- Configurable endpoint URL

### Rate Limiting

- Default delay between requests: 1.0 second
- Configurable via `EvaluationConfig.rate_limit_delay`

### Retry Logic

- Max retries: 3
- Exponential backoff: 2s, 4s, 8s

## Local LLaVA Setup

For running LLaVA locally without API costs:

### Option 1: llama.cpp Server

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. && cmake --build . --config Release

# Download model
wget https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/ggml-model-q4_k.gguf
wget https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/mmproj-model-f16.gguf

# Start server
./llama-server -m ggml-model-q4_k.gguf --mmproj mmproj-model-f16.gguf --port 8080
```

### Option 2: HuggingFace Transformers

See `vision_model_evaluator_notebook.ipynb` for GPU-based deployment using transformers.

## Examples

### Basic Traffic Sign Validation

```bash
python vision_model_evaluator.py \
  --model gpt4v \
  --api-key $OPENAI_API_KEY \
  --input ./traffic_signs \
  --output ./results/validation_run
```

### Comparing Models

```bash
# Run GPT-4V evaluation
python vision_model_evaluator.py --model gpt4v --api-key $OPENAI_API_KEY \
  --input ./images --output ./results/gpt4v

# Run Gemini evaluation
python vision_model_evaluator.py --model gemini --api-key $GEMINI_API_KEY \
  --input ./images --output ./results/gemini

# Compare results
python -c "
import pandas as pd
gpt4v = pd.read_csv('./results/gpt4v.csv')
gemini = pd.read_csv('./results/gemini.csv')
print('GPT-4V Accuracy:', (gpt4v['prediction'] == 'yes').mean())
print('Gemini Accuracy:', (gemini['prediction'] == 'yes').mean())
"
```

### Custom Classification Task

```bash
python vision_model_evaluator.py \
  --model gpt4v \
  --api-key $OPENAI_API_KEY \
  --input ./images \
  --output ./results/classification \
  --prompt "Classify this object as: vehicle, pedestrian, or infrastructure. Answer with one word."
```

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors:
- Increase `rate_limit_delay` in the code (default: 1.0s)
- Use a higher-tier API plan with increased quotas
- Process images in smaller batches

### Timeout Errors

If requests timeout frequently:
- Increase `timeout` in `ModelConfig` (default: 60.0s)
- Check your internet connection
- Verify API service status

### Prediction Extraction Issues

If predictions show "unclear":
- Review `raw_response` column in CSV output
- Adjust the prompt to request more explicit yes/no answers
- Customize `_extract_prediction()` method for your use case

## Contributing

Contributions are welcome! To add support for new vision models:

1. Create a new client class inheriting from `VisionModelClient`
2. Implement the `analyze_image()` method
3. Add the model to the `ModelType` enum
4. Update the factory function in `create_client()`
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Google Gemini API
- OpenAI GPT-4 Vision API
- LLaVA: Large Language and Vision Assistant
- llama.cpp for efficient local inference
