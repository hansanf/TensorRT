def dump_csv(data, csv_txt_path):
    """输出一致性txt表格

    Args:
        data (_type_): _description_
        csv_txt_path (_type_): _description_
    """
    # 计算每列的最大宽度
    col_widths = [max(len(str(x)) for x in col) for col in zip(*data)]

    # 打开文件并写入表格
    with open(csv_txt_path, 'w') as file:
        for i, row in enumerate(data):
            # 使用字符串格式化输出每一行
            formatted_row = ' │ '.join(f'{str(item):<{width}}' for item, width in zip(row, col_widths))
            formatted_row_ = '─┼─'.join(f'{"─"*width}' for item, width in zip(row, col_widths))
            formatted_row_ = '├' + formatted_row_ + '┤'
            formatted_row = '│' + formatted_row + '│'
            file.write(formatted_row + '\n')
            file.write(formatted_row_ + '\n')
            
            
            
data = [['11', '22', 'abc']]
dump_csv(data, "1.csv")