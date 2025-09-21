import pandas as pd
import os
import sys
from argparse import ArgumentParser

def merge_excel_sheets(file_path, output_path):
    # 读取所有 sheet
    xls = pd.ExcelFile(file_path)
    merged_df = pd.DataFrame()

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 如果 sheet 只有一列，没有列名，就用 sheet 名作为列名
        if df.shape[1] == 1 and df.columns[0] == 0:
            df.columns = [sheet_name]
        else:
            # 防止重名，前面加 sheet 名
            df = df.add_prefix(sheet_name + "_")

        # 横向拼接
        merged_df = pd.concat([merged_df, df], axis=1)

    merged_df.to_excel(output_path, index=False)
    print(f"Merged sheets saved to {output_path}")

def batch_merge_excels_recursive(root_dir):
    """递归处理目录下所有 Excel 文件"""
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls"):
                file_path = os.path.join(folder, file)
                output_path = os.path.join(folder, f"merged_{file}")
                try:
                    merge_excel_sheets(file_path, output_path)
                except Exception as e:
                    print(f"Failed to merge {file_path}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="script parameters")
    parser.add_argument("--input_path", type=str, required=True, help="Parent directory containing .mat files")
    args = parser.parse_args(sys.argv[1:])

    batch_merge_excels_recursive(args.input_path)
    print(f"Merged Excel files in {args.input_path}, successfully.")
