import subprocess

def run_ollama_prompt(model_name, prompt_text):
    try:
        command = ['ollama', 'run', model_name]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=prompt_text)

        if process.returncode != 0:
            print(f"❌ Error: {stderr}")
            return None

        return stdout.strip()

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return None


# Example usage
model = "llama3:latest"
prompt = "do you know the meaning of life? Please answer in JSON format with a key 'answer'."

response = run_ollama_prompt(model, prompt)
print(response)