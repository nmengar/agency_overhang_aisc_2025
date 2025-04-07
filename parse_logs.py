import json

if __name__ == '__main__':
    with open('./logs/2025-03-21T23-57-56-07-00_hello-world_RJrrTZ6kkP73BfjYbJUB9N.json','r') as file:
        data = json.load(file)
    outputs = {}
    for sample in data['samples']:
        outputs[sample['input']] = {
            'enhanced_prompt': sample['messages'][1]['content'] # The assistant response
        }
    with open('./enhanced_prompts/hello_world1.json','w+') as file:
        json.dump(outputs, file, indent=4)
    