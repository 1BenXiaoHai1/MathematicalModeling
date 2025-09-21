import scipy.io as sio
import h5py
import pandas as pd
from argparse import ArgumentParser, Namespace
import sys
import os
import glob

def read_mat_file(file_path):
    try:
        # 尝试用 scipy 读取 (v7.2 以下)
        data = sio.loadmat(file_path)
        # 删除一些无关的键
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        return data
    except NotImplementedError:
        # 如果是 v7.3（基于 HDF5）
        with h5py.File(file_path, "r") as f:
            data = {}
            for key in f.keys():
                data[key] = f[key][:]
        return data

def mat_to_excel(mat_file, excel_file):
    data = read_mat_file(mat_file)
    writer = pd.ExcelWriter(excel_file, engine="openpyxl")

    for var_name, values in data.items():
        try:
            df = pd.DataFrame(values)
        except Exception:
            # 如果不是二维数组，转成一维列表保存
            df = pd.DataFrame(values.flatten())
        df.to_excel(writer, sheet_name=var_name[:31], index=False)  # Excel sheet名最长31
    writer.close()

def batch_convert_mat_to_excel(input_dir, output_dir):
    """批量转换目录下所有 .mat 文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))
    if not mat_files:
        print("There are no .mat files in the specified directory.")
        return

    for mat_file in mat_files:
        file_name = os.path.splitext(os.path.basename(mat_file))[0]
        excel_file = os.path.join(output_dir, file_name + ".xlsx")
        try:
            mat_to_excel(mat_file, excel_file)
            print(f" Successfully converted: {mat_file} to {excel_file}")
        except Exception as e:
            print(f" Failed to convert {mat_file}: {e}")

def convert_all_mat_in_folder(root_dir):
    """递归扫描 root_dir 下所有 .mat 文件并转换为同名 .xlsx"""
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mat"):
                mat_path = os.path.join(folder, file)
                excel_path = os.path.splitext(mat_path)[0] + ".xlsx"
                try:
                    mat_to_excel(mat_path, excel_path)
                    print(f"Successfully converted: {mat_path} to {excel_path}")
                except Exception as e:
                    print(f"Failed to convert {mat_path}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="script parameters")
    parser.add_argument("--input_path", type=str, required=True, help="Parent directory containing .mat files")
    # parser.add_argument("--output_path", type=str, required=True, help="Parent directory to save .xlsx files")
    args = parser.parse_args(sys.argv[1:])

    # batch_convert_mat_to_excel(args.input_path, args.output_path)
    convert_all_mat_in_folder(args.input_path)
    print(f"Converted {args.input_path}, successfully.")
