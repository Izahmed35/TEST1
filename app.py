import os
import replicate
from flask import Flask, render_template, request, jsonify
from pydantic import BaseModel
from typing import Optional, Any

app = Flask(__name__)

# Ensure you have set the REPLICATE_API_TOKEN environment variable
replicate.api_token = os.environ.get("REPLICATE_API_TOKEN")

class CustomPrediction(BaseModel):
    id: str
    version: str
    urls: dict
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str
    input: dict
    output: Any
    error: Optional[str] = None
    logs: str
    metrics: dict

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        try:
            output = replicate.run(
                "izahmed35/izmodel:0e9f080f7e29f6800cf3ba745587fdf21824d33184c8f5264976fddfa02d135c",
                input={
                    "prompt": prompt,
                    "model": "dev",
                    "lora_scale": 1,
                    "num_outputs": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "webp",
                    "guidance_scale": 3.5,
                    "output_quality": 90,
                    "prompt_strength": 0.8,
                    "extra_lora_scale": 1,
                    "num_inference_steps": 28
                }
            )
            
            # Convert the output to our custom prediction model
            prediction = CustomPrediction(**output)
            
            # Check if output is available
            if prediction.output and isinstance(prediction.output, list) and len(prediction.output) > 0:
                image_url = prediction.output[0]
                return jsonify({"image_url": image_url})
            else:
                return jsonify({"error": "No image URL returned from the model"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
