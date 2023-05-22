import argparse
import uuid
import pandas as pd
from tqdm.auto import tqdm
from azure.storage.blob import BlobServiceClient
from os.path import join


def uploadToBlobStorage(file_path, file_name, connection_string, container_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
    with open(file_path, 'rb') as data:
        blob_client.upload_blob(data)


def main(args):
    df = pd.read_csv(args.prompts_file)
    lines = []
    cur_idx = 0

    for i, row in df.iterrows():
        for j in range(4):
            img_name = f'{str(uuid.uuid1())}.png'
            lines.append([cur_idx, row['prompt'], row['image_description'], img_name, j])
            cur_idx += 1
    res_map = pd.DataFrame(lines, columns=['img_idx', 'prompt', 'image_description', 'image_name', 'img_id_in_prompt'])
    res_map.to_csv(args.img_name_map_file, index=None)

    cur_idx = 0
    for prompt_id in tqdm(range(0, 19689)):
        for j in range(4):
            img_name = res_map.loc[cur_idx]['image_name']
            uploadToBlobStorage(join(args.img_dir, f'{cur_idx}.png'), img_name, args.connection_string, args.container_name)
            cur_idx += 1

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, default='gpt2_prompts.csv')
    parser.add_argument('--img_name_map_file', type=str, default='img_name_map.csv')
    parser.add_argument('--img_dir', type=str, default='imgs')
    parser.add_argument('--connection_string', type=str, required=True, help='Azure connection string')
    parser.add_argument('--container_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
