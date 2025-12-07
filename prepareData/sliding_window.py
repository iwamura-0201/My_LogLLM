from pathlib import Path


import numpy as np
import pandas as pd
from .helper import sliding_window, fixedSize_window, structure_log

#### for Thunderbird, Liberty, BGL（ログデータが時系列順＆セッションID等を持たない）


data_dir = r'/mnt/public/gw/SyslogData/BGL'
log_name = "BGL.log"

start_line = 0
end_line = None

# # Liberty
# start_line = 40000000
# end_line = 45000000

# # thunderbird
# start_line = 160000000
# end_line = 170000000

output_dir = data_dir



def main(
    data_dir: Path,
    log_filename: str,
    output_dir: Path,
    group_type: str = "fixedSize",
    start_line: int = 0,
    end_line: int = None,
):
    # group_type = 'time_sliding'

    window_size = 100
    step_size = 100

    # if 'bgl' in log_name.lower():
    #     window_size = 5  # 5 minutes
    #     step_size = 1  # 1  minutes
    # elif 'thunderbird' in log_name.lower():
    #     window_size = 1 # 1 minutes
    #     step_size = 0.5 # 0.5  minutes
    # else:
    #     raise Exception('missing valid window_size and step_size')

    if 'thunderbird' in log_filename.lower() or 'spirit' in log_filename.lower() or 'liberty' in log_filename.lower():
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'   #thunderbird  , spirit, liberty
    elif 'bgl' in log_filename.lower():
        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'  #bgl
    elif 'security' in log_filename.lower():
        log_format = "<Version>,<Computer>,<Execution_ThreadID>,<Channel>,<Content>,<Provider_Name>,<Correlation_RelatedActivityID>,<Keywords>,<Opcode>,<Correlation_ActivityID>,<Execution_ProcessID>,<Security_UserID>,<Task>,<Level>,<Provider_Guid>,<TimeCreated_SystemTime>,<EventRecordID>,<EventID>,<project>"  #security
    else:
        raise Exception('missing valid log format')
    print(f'Auto log_format: {log_format}')

    # 一旦Drainによるパースをそのまま利用（12/8時点）
    # structure_log(data_dir, output_dir, log_name, log_format, start_line = start_line, end_line = end_line)

    # ----------------------------- ここからwindow分割処理 ---------------------------------#

    print(f'window_size: {window_size}; step_size: {step_size}')

    train_ratio = 0.8

    df = pd.read_csv(data_dir/f'{log_filename}_structured.csv')

    print(len(df))

    # data preprocess
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    if group_type == 'time_sliding':
        df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0)

    train_len = int(train_ratio*len(df))

    df_train = df[:train_len]

    df_test = df[train_len:]
    df_test = df_test.reset_index(drop=True)

    print('Start grouping.')

    # 元コードでは fixed 利用推奨？

    if group_type == 'time_sliding':
        # grouping with time sliding window
        session_train_df = sliding_window(df_train[["timestamp", "Label", "deltaT",'Content']],
                                    para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60}
                                    )

        # grouping with time sliding window
        session_test_df = sliding_window(df_test[["timestamp", "Label", "deltaT",'Content']],
                                    para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60}
                                    )
    elif group_type == 'fixedSize':
        # grouping with fixedSize window
        session_train_df = fixedSize_window(
            df_train[['Content', 'Label']],
            window_size = window_size, step_size = step_size
            )

        # grouping with fixedSize window
        session_test_df = fixedSize_window(
            df_test[['Content', 'Label']],
            window_size = window_size, step_size = step_size
            )
    else:
        raise Exception('missing valid group_type')

    # ---------------------------------- ここからデータ整形 -----------------------------------#

    col = ['Content', 'Label','item_Label']
    spliter=' ;-; '

    session_train_df = session_train_df[col]
    session_train_df['session_length'] = session_train_df["Content"].apply(len)
    session_train_df["Content"] = session_train_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_train_len = session_train_df['session_length'].mean()
    max_session_train_len = session_train_df['session_length'].max()
    num_anomalous_train= session_train_df['Label'].sum()
    num_normal_train = len(session_train_df['Label']) - session_train_df['Label'].sum()

    session_test_df = session_test_df[col]
    session_test_df['session_length'] = session_test_df["Content"].apply(len)
    session_test_df["Content"] = session_test_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_test_len = session_test_df['session_length'].mean()
    max_session_test_len = session_test_df['session_length'].max()
    num_anomalous_test= session_test_df['Label'].sum()
    num_normal_test = len(session_test_df['Label']) - session_test_df['Label'].sum()


    session_train_df.to_csv(os.path.join(output_dir, 'train.csv'),index=False)
    session_test_df.to_csv(os.path.join(output_dir, 'test.csv'),index=False)

    print('Train dataset info:')
    print(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
    print(f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    print('Test dataset info:')
    print(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
    print(f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")

    output_file = output_dir / 'train_info.txt'
    with output_file.open('w') as file:
        # ファイルに内容を書き込み
        file.write(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
        file.write(f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    output_file = output_dir / 'test_info.txt'
    with output_file.open('w') as file:
        # ファイルに内容を書き込み
        file.write(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
        file.write(f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")


if __name__ == '__main__':
    main()
