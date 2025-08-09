import numpy as np
from transformers import AutoConfig
import sglang as sgl
from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

# Initialize SGLang components from test.py
@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen("action")


class TokenToAction:
    def __init__(self, n_action_bins: int = 256, unnorm_key: str = "bridge_orig"):
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = 32000
        self.unnorm_key = unnorm_key
        self.config = AutoConfig.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        ).to_dict()
        self.norm_stats = self.config["norm_stats"]
        assert unnorm_key is not None
        if unnorm_key not in self.norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {self.norm_stats.keys()}"
            )

    def convert(self, output_ids):
        predicted_action_token_ids = np.array(output_ids)
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions


# Global converter instance
converter = TokenToAction()

# Global runtime (will be initialized on startup)
runtime = None


@app.route('/batch', methods=['POST'])
def batch_endpoint():
    """
    Process batch of instructions and return output_ids and actions.
    Compatible with quickstart.py expectations.
    """
    try:
        data = request.get_json()
        
        # Extract parameters from request
        instructions = data.get('instructions', [])
        image_path = data.get('image_path', 'robot.jpg')
        temperature = data.get('temperature', 0.0)
        
        if not instructions:
            return jsonify({'error': 'No instructions provided'}), 400
        
        # Ensure image path is absolute
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)
        
        # Check if image exists
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file not found: {image_path}'}), 400
        
        # Prepare batch arguments for SGLang
        arguments = []
        for instruction in instructions:
            # Format question similar to test.py
            question = f"In: What action should the robot take to {instruction}?\nOut:"
            arguments.append({
                "image_path": image_path,
                "question": question
            })
        
        # Process batch using SGLang (from test.py logic)
        states = image_qa.run_batch(
            arguments,
            max_new_tokens=7,
            temperature=temperature,
            return_logprob=True
        )
        
        # Extract results
        all_output_ids = []
        all_actions = []
        
        for state in states:
            # Get output token IDs (from test.py logic)
            output_logprobs = state.get_meta_info("action")["output_token_logprobs"]
            output_ids = [logprob[1] for logprob in output_logprobs]
            
            # Convert to actions using TokenToAction converter
            actions = converter.convert(output_ids)
            
            all_output_ids.append(output_ids)
            all_actions.append(actions.tolist())
        
        # Return results in format expected by quickstart.py
        response = {
            'output_ids': all_output_ids,
            'actions': all_actions
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/single', methods=['POST'])
def single_endpoint():
    """
    Process single instruction (convenience endpoint).
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        instruction = data.get('instruction', '')
        image_path = data.get('image_path', 'robot.jpg')
        temperature = data.get('temperature', 0.0)
        
        if not instruction:
            return jsonify({'error': 'No instruction provided'}), 400
        
        # Use batch endpoint with single instruction
        batch_data = {
            'instructions': [instruction],
            'image_path': image_path,
            'temperature': temperature
        }
        
        # Process using batch logic
        response = batch_endpoint()
        if response.status_code != 200:
            return response
        
        # Extract single result
        result_data = response.get_json()
        single_result = {
            'output_ids': result_data['output_ids'][0],
            'actions': result_data['actions'][0]
        }
        
        return jsonify(single_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({'status': 'healthy', 'model': 'openvla-7b'})


def initialize_runtime():
    """
    Initialize SGLang runtime (from test.py setup).
    """
    global runtime
    
    print("Initializing SGLang runtime...")
    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_radix_cache=True,
        trust_remote_code=True,
    )
    sgl.set_default_backend(runtime)
    print("SGLang runtime initialized successfully!")


def shutdown_runtime():
    """
    Shutdown SGLang runtime.
    """
    global runtime
    if runtime:
        runtime.shutdown()
        print("SGLang runtime shutdown.")


if __name__ == '__main__':
    try:
        # Initialize the runtime before starting the server
        initialize_runtime()
        
        print("Starting API server on localhost:3200...")
        print("Compatible with quickstart.py")
        print("Endpoints:")
        print("  POST /batch - Process batch of instructions")
        print("  POST /single - Process single instruction") 
        print("  GET /health - Health check")
        
        # Start Flask server
        app.run(host='0.0.0.0', port=3200, debug=False)
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        shutdown_runtime()
