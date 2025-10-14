import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from percept_parser import plotter
import os

class PerceptParser:

    def __init__(self, filename: str, verbose: bool = False):
        self.verbose = verbose

        with open(filename, 'r') as f:
            self.js = json.load(f)

        self.session_date = datetime.strptime(self.js['SessionDate'], '%Y-%m-%dT%H:%M:%SZ')
        self.lead_location = self.js['LeadConfiguration']['Final'][0]['LeadLocation'].split('.')[-1].upper()
        self.lead_model = self.js['LeadConfiguration']['Final'][0]['Model'].split('.')[-1].upper()
        self.subject = self.js["PatientInformation"]["Final"]["PatientId"]
        self.diagnosis = self.js["PatientInformation"]["Final"]["Diagnosis"]
        print(f"{filename}: {self.session_date} - {self.lead_location}")

    def parse_all(self, out_path: str = "sub"):

        os.makedirs(out_path, exist_ok=True)

        df_lfp_trend_logs = self.parse_lfp_trend_logs()
        df_brainsense_lfp = self.parse_brain_sense_lfp()
        dfs_bs_td = self.read_timedomain_data(indefinite_streaming=False)
        dfs_is_td = self.read_timedomain_data(indefinite_streaming=True)

        if df_lfp_trend_logs.empty is False:
            df_lfp_trend_logs.to_csv(os.path.join(out_path, "LFPTrendLogs.csv"), index=True)
            plotter.lfptrendlog_plot(df_lfp_trend_logs, self, path_out=out_path)
        
        if df_brainsense_lfp.empty is False:
            df_brainsense_lfp.to_csv(os.path.join(out_path, "BrainSenseLfp.csv"), index=True)
            plotter.brain_sense_lfp_plot(df_brainsense_lfp, self, out_path=out_path)
        
        if len(dfs_bs_td) > 0:
            plotter.plot_time_domain_ranges(dfs_bs_td, out_path=out_path)
            for _, df_bs_td_i in tqdm(list(enumerate(dfs_bs_td)), desc="BrainSenseTimeDomain Plot Index"):
                str_idx = df_bs_td_i.index[0].strftime("%Y-%m-%d %H:%M:%S") + f" - {df_bs_td_i.index[-1].strftime('%H:%M:%S')}"
                df_bs_td_i.to_csv(os.path.join(out_path, f"BrainSenseTimeDomain_{str_idx}.csv"), index=True)
                plotter.plot_df_timeseries(df_bs_td_i, out_path=out_path)
                plotter.time_frequency_plot_td(df_bs_td_i, indefinite_streaming=False, parser=self, out_path=out_path)
        
        if len(dfs_is_td) > 0:
            plotter.plot_time_domain_ranges(dfs_is_td, out_path=out_path)
            for _, df_is_td_i in tqdm(list(enumerate(dfs_is_td)), desc="IndefiniteStreaming Plot Index"):
                str_idx = df_is_td_i.index[0].strftime("%Y-%m-%d %H:%M:%S") + f" - {df_is_td_i.index[-1].strftime('%H:%M:%S')}"
                df_is_td_i.to_csv(os.path.join(out_path, f"IndefiniteStreaming_{str_idx}.csv"), index=True)
                plotter.plot_df_timeseries(df_is_td_i, out_path=out_path)
                plotter.time_frequency_plot_td(df_bs_td_i, indefinite_streaming=True, parser=self, out_path=out_path)

        

    def parse_lfp_trend_logs(self, ):
        df_idx = []
        if "LFPTrendLogs" not in self.js["DiagnosticData"]:
            print("No LFPTrendLogs found in the JSON file.")
            return pd.DataFrame()
        LFPTrendLogs = self.js["DiagnosticData"]["LFPTrendLogs"]
        for hemisphere in ["Left", "Right"]:
            if hemisphere == "Left":
                hem_name = "HemisphereLocationDef.Right"
            else:
                hem_name = "HemisphereLocationDef.Left"
            for date_time_start in LFPTrendLogs[hem_name].keys():
                for date_time_idx in range(len(LFPTrendLogs[hem_name][date_time_start])):
                    date = LFPTrendLogs[hem_name][date_time_start][date_time_idx]['DateTime']
                    lfp = LFPTrendLogs[hem_name][date_time_start][date_time_idx]['LFP']
                    mA = LFPTrendLogs[hem_name][date_time_start][date_time_idx]['AmplitudeInMilliAmps']
                    df_idx.append({
                        "Time": pd.to_datetime(date),
                        "LFP": lfp,
                        "mA": mA,
                        "Hemisphere": hemisphere,
                    })
        df = pd.DataFrame(df_idx)
        df = df.set_index("Time")
        df = df.sort_index()
        return df

    def parse_brain_sense_lfp(self):
        df_idx = []
        if "BrainSenseLfp" not in self.js:
            print("No BrainSenseLfp found in the JSON file.")
            return pd.DataFrame()
        for index_ in range(len(self.js["BrainSenseLfp"])):
            js_td = self.js["BrainSenseLfp"][index_]

            FirstPacketDateTime = js_td['FirstPacketDateTime']
            sfreq = js_td['SampleRateInHz']
            LFPData = js_td['LfpData']
            dat_ = []
            for index_lfp in range(len(LFPData)):
                TicksInMses = LFPData[index_lfp]["TicksInMs"]
                Right_v = LFPData[index_lfp]["Right"]["LFP"]
                Left_v = LFPData[index_lfp]["Left"]["LFP"]
                Right_c = LFPData[index_lfp]["Right"]["mA"]
                Left_c = LFPData[index_lfp]["Left"]["mA"]
                dat_.append({
                    #"FirstPacketDateTime": FirstPacketDateTime,
                    "SampleRateInHz": sfreq,
                    "TicksInMses": TicksInMses,
                    "Right_power": Right_v,
                    "Left_power": Left_v,
                    "Right_stim_current": Right_c,
                    "Left_stim_current": Left_c
                })

            dat_lfp = pd.DataFrame(dat_)
            dat_lfp["TicksInMsesDiff"] = dat_lfp["TicksInMses"].diff().fillna(0)
            dat_lfp["TicksInMsesDiff_cumul"] = dat_lfp["TicksInMsesDiff"].cumsum()
            dat_lfp["Time"] = pd.to_datetime(FirstPacketDateTime) + pd.to_timedelta(dat_lfp["TicksInMsesDiff_cumul"], unit='ms')
            df_idx.append(dat_lfp[["Time", "Right_power", "Left_power", "Right_stim_current", "Left_stim_current"]])

        df = pd.concat(df_idx, ignore_index=True)
        df = df.set_index("Time")
        df = df.sort_index()
        #df = df.resample(f"{1/sfreq:.6f}S").mean()
        return df

    def get_time_stream(
        self, data: dict, num_chs: int, verbose: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
        """
        Process a Streaming Sample object from the Percept JSON data
        """

        start_time = datetime.fromisoformat(data["FirstPacketDateTime"])
        fs = data["SampleRateInHz"]

        TimeDomainData = np.array(data["TimeDomainData"])
        GlobalPacketSizes = np.fromstring(data["GlobalPacketSizes"], sep=",", dtype=int)

        if np.sum(GlobalPacketSizes) != TimeDomainData.shape[0]:
            raise ValueError("GlobalPacketSizes does not match TimeDomainData length")

        GlobalSequences = np.fromstring(data["GlobalSequences"], sep=",", dtype=int)
        TicksInMses = np.fromstring(data["TicksInMses"], sep=",", dtype=int)
        TicksDiff = np.insert(np.diff(TicksInMses), 0, 0)

        # Check that packages are sorteds
        if not np.all(TicksInMses[:-1] <= TicksInMses[1:]):
            raise ValueError(f"Packages in the wrong order")

        num_packets = len(GlobalPacketSizes)
        PacketLoss = TicksDiff / 1000 > (GlobalPacketSizes + 1) / fs

        # This works without accounting for errors
        RelativeTimes = (np.cumsum(GlobalPacketSizes - 1) + np.arange(num_packets)) / fs

        # if indefinite_streaming:
        #     # collapse df_i
        #     idx_250_before = df_i.query("TicksInMsesDiff == 250").index-1


        un, counts = np.unique(TicksDiff, return_counts=True)
        df_counts = pd.DataFrame({'TicksDiff': un, 'Counts': counts})

        # Get relative sample times for the first packet
        tdtime = np.arange(0, GlobalPacketSizes[0]) / fs

        t_cur = tdtime[-1] + (1 / fs)  # First timepoint of next package

        last_times = [tdtime[-1]]

        l_tdtime = []
        l_tdtime.append(tdtime)
        PACKAGE_LOSS_PRESENT = False

        for i in range(1, num_packets):
            packet_size = GlobalPacketSizes[i]
            time_diff = TicksDiff[i] / 1000  # s

            # for indefinite streaming with 6 chs diffs are 250
            # for brainsense time domain data there are two chs
            
            #if indefinite_streaming:
            #    time_diff = time_diff / 2  # split in sequence of half

            # if there are two channels, don't adapt time_diff
            if num_chs == 3:
                time_diff = time_diff / 3  # here I don't know, 
            elif num_chs == 6:
                time_diff = time_diff / 2

            # Is this even possible? If so, would need to reorder TimeDomainData
            # if time_diff < 0:
            #     time_diff = np.bincount(TicksDiff).argmax() / 1000  # most freq value

            # Get relative sample times for packet
            td_time_packet = np.arange(0, packet_size) / fs

            # t_cur is first time of this packet
            if time_diff > (packet_size + 1) / fs:  # If packages missing
                PACKAGE_LOSS_PRESENT = True
                # To get t_cur to first time of packet after adjusting for missing packets:
                # 1. Revert t_cur to end of previous of package
                # 2. Advance by time diff between packets
                # 3. Go back by the duration of this packet
                # td_time_packet += (t_cur - 1 / fs) + time_diff - td_time_packet[-1]
                # Note: this method is risky because time_diff is rounded to the 50ms
                # Documentation suggests just using the TicksInMses values
                td_time_packet += (
                    ((TicksInMses[i] - TicksInMses[0]) / 1000)
                    + (GlobalPacketSizes[0] - 1) / fs
                    - (packet_size - 1) / fs
                )
            else:  # If not missed packages
                # print(t_cur + td_time_packet[-1]) # last absolute time of packet
                td_time_packet += t_cur
                # print(RelativeTimes[i])

            last_times.append(td_time_packet[-1])
            t_cur = td_time_packet[-1] + (1 / fs)  # move to first time of next packet
            l_tdtime.append(td_time_packet)
            tdtime = np.concatenate([tdtime, td_time_packet])

        # Check that calculation was correct by comapring to actual last value from TicksInMses
        print(
            f"Last value: {
                (TicksInMses[-1] - TicksInMses[0]) / 1000
                + (GlobalPacketSizes[0] - 1) / fs
            }"
        )

        if tdtime.shape[0] != TimeDomainData.shape[0]:
            raise ValueError(f"tdtime shape {tdtime.shape} does not match TimeDomainData shape {TimeDomainData.shape}") 

        td_ = pd.to_timedelta(tdtime, unit="s") + pd.Timestamp(start_time)
        ch_ = data["Channel"]
        df_ch = pd.DataFrame(
            {
                "Time": td_,
                "Data": TimeDomainData,
            }
        )
        df_ch = df_ch.set_index("Time")

        df_ch = df_ch.resample(f"{int(1000/fs)}ms").mean()
        df_ch["Channel"] = ch_

        if verbose:
            from matplotlib import pyplot as plt
            plt.subplot(1, 2, 1)
            plt.plot(df_ch.query("Channel == @ch_")["Data"].iloc[-500:].values)
            plt.title(f"Corrected Channel {ch_} - TimeDomainData")
            plt.subplot(1, 2, 2)
            plt.plot(TimeDomainData[-500:])
            plt.title(f"JSON TimeDomainData - Channel {ch_}")
            plt.tight_layout()
            plt.xlabel("Samples")

        return df_ch, df_counts, PACKAGE_LOSS_PRESENT

    def read_timedomain_data(self, indefinite_streaming: bool = True) -> pd.DataFrame:

        if indefinite_streaming:
            str_timedomain = "IndefiniteStreaming"
        else:
            str_timedomain = "BrainSenseTimeDomain"
        if str_timedomain not in self.js:
            print(f"No {str_timedomain} found in the JSON file.")
            return []

        td_data = self.js[str_timedomain]

        # Each channel has its own "Streaming Sample" data structure
        # We can identify each recording session by the first data packet timestamp
        first_packet_times = np.unique(
            [stream["FirstPacketDateTime"] for stream in td_data]
        )

        channels = np.unique([stream["Channel"] for stream in td_data])

        df_ = []
        for package_idx, first_package in tqdm(list(enumerate(np.unique(first_packet_times))),
                                               desc=f"{str_timedomain} Index"):
            df_counts_sum = []
            df_chs = []
            for session_idx in np.where(first_packet_times == first_package)[0]:
                df_ch, df_counts, PACKAGE_LOSS_PRESENT = self.get_time_stream(
                    data=td_data[session_idx],
                    num_chs=len(channels),  # Why is this param needed?
                    verbose=False,
                )
                df_chs.append(df_ch)
                # These are currently unused
                # df_counts["file_idx"] = session_idx
                # df_counts_sum.append(df_counts)

            # Alternatively
            # df_ch = [
            #     self.get_time_stream(
            #         js_td=td_data[session_idx],
            #         num_chs=len(channels),  # Why is this param needed?
            #         verbose=False,
            #     )[0]
            #     for session_idx in np.where(first_packet_times == first_datetime)[0]
            # ]

            df_concat = pd.concat(df_chs, axis=0)
            df_concat = df_concat.reset_index().pivot(index='Time', columns='Channel', values='Data')
            df_.append(df_concat)

        return df_

    # def read_brainsense_timedomain(self, ) -> pd.DataFrame:

    #     if "BrainSenseTimeDomain" not in self.js:
    #         print("No BrainSenseTimeDomain found in the JSON file.")
    #         return pd.DataFrame()

    #     df_counts_sum = []
    #     df_chs = []
    #     for index_ in tqdm(range(len(self.js["BrainSenseTimeDomain"])), desc="BSTimeDomain Index"):
    #         df_ch, df_counts, PACKAGE_LOSS_PRESENT = self.get_time_stream(self.js["BrainSenseTimeDomain"][index_], False, indefinite_streaming=False)
    #         df_counts["file_idx"] = index_
    #         df_counts_sum.append(df_counts)
            
    #         df_chs.append(df_ch)
    #     df_concat = pd.concat(df_chs, axis=0).reset_index()
    #     df_concat = df_concat.drop_duplicates(subset=['Time', 'Channel'])
    #     df_concat = df_concat.reset_index().pivot(index='Time', columns='Channel', values='Data')

    #     df_counts = pd.concat(df_counts_sum, axis=0)
    #     df_counts["Time_Diff_Total"] = df_counts["Counts"] * df_counts["TicksDiff"]
    #     file_sum = df_counts.groupby("file_idx")["Time_Diff_Total"].sum().reset_index()
    #     file_missing = df_counts.query("TicksDiff != 250").groupby("file_idx")["Time_Diff_Total"].sum().reset_index()

    #     df_counts_file_idx = pd.merge(file_sum, file_missing, on="file_idx", how="outer", suffixes=('_sum', '_missing'))
    #     df_counts_file_idx = df_counts_file_idx.fillna(0)
    #     df_counts_file_idx = df_counts_file_idx.sort_values(by="Time_Diff_Total_missing", ascending=True).iloc[::2]


    #     df_counts_file_idx["Time_Diff_Total_sum"] = df_counts_file_idx["Time_Diff_Total_sum"] / 1000
    #     df_counts_file_idx["Time_Diff_Total_missing"] = df_counts_file_idx["Time_Diff_Total_missing"] / 1000

    #     df_counts_file_idx["Time_Diff_Total_missing_clipped"] = df_counts_file_idx["Time_Diff_Total_missing"].clip(upper=10)

    #     return df_concat, df_counts_file_idx
