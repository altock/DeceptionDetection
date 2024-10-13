import json


def convert_to_openai_format(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)

            # Extract the user and assistant messages
            user_message = data["input"][0]["content"]
            assistant_message = data["input"][1]["content"]

            # Create the OpenAI format
            openai_format = {
                "messages": [
                    {"role": "user", "content": user_message},
                    {
                        "role": "assistant",
                        "content": assistant_message
                        + (" 😉" if data["target"][0] == "😉" else ""),
                    },
                ]
            }

            # Write the formatted data to the output file
            json.dump(openai_format, outfile)
            outfile.write("\n")


# Usage
input_file = "input_files/gpt4o_misleading_large.jsonl"
output_file = "finetune_files/gpt4o_misleading_large_finetune.jsonl"
convert_to_openai_format(input_file, output_file)
