import fire
import pandas as pd
import os 
def main(
    document_dir = '',
):
    for document_fpath in os.listdir(document_dir):
        if document_fpath.endswith('.parquet'):
            df_docs = pd.read_parquet(os.path.join(document_dir, document_fpath))
            df_docs = df_docs.sort_values(by='q_idx').reset_index(drop=True)
            df_docs.to_parquet(os.path.join(document_dir, document_fpath), index=False)

if __name__ == '__main__':
    fire.Fire(main)