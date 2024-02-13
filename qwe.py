
def transform_csv(in_file, cols, std_vals, norm_vals):
    df = pd.read_csv(in_file)
    df_2 = df.apply(lambda col: col if col.name not in cols else (col - std_vals[col.name]['mean']) / std_vals[col.name]['std'])
    file_name, extension = in_file.split(".")
    df_2.to_csv(f'{file_name}_standardized.{extension}', index=False)
    df_3 = df.apply(lambda col: col if col.name not in cols else (col - norm_vals[col.name]['min']) / (norm_vals[col.name]['max'] - norm_vals[col.name]['min']))
    df_3.to_csv(f'{file_name}_normalized.{extension}', index=False)
