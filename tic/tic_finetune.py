import openai
import wandb
import os
import time

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize wandb
wandb.init(project="gpt4o-lie-finetune", job_type="fine-tuning")

# Specify the path to your fine-tuning file
file_path = "finetune_files/gpt4o_misleading_large_finetune.jsonl"

# Upload the file to OpenAI
upload_response = openai.File.create(file=open(file_path, "rb"), purpose="fine-tune")
file_id = upload_response.id

print(f"File uploaded successfully. File ID: {file_id}")

# Create a fine-tuning job
job_response = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-4o-mini",  # You can change this to the appropriate base model
    hyperparameters={
        "n_epochs": 3,  # Adjust as needed
    },
)

job_id = job_response.id
print(f"Fine-tuning job created. Job ID: {job_id}")

# Monitor the fine-tuning job
while True:
    job_status = openai.FineTuningJob.retrieve(job_id)
    print(f"Job status: {job_status.status}")

    if job_status.status in ["succeeded", "failed"]:
        break

    # Log metrics to wandb
    if hasattr(job_status, "result"):
        wandb.log(
            {
                "training_loss": job_status.result.training_loss,
                "elapsed_tokens": job_status.result.trained_tokens,
            }
        )

    time.sleep(60)  # Check status every minute

# Log final results
if job_status.status == "succeeded":
    print("Fine-tuning completed successfully!")
    print(f"Fine-tuned model: {job_status.fine_tuned_model}")
    wandb.log({"fine_tuned_model": job_status.fine_tuned_model})
else:
    print("Fine-tuning failed.")
    wandb.log({"status": "failed"})

# Finish wandb run
wandb.finish()
