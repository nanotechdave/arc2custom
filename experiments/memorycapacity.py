import os
import sys
import time
from datetime import date

import numpy as np
import pyarc2

import dparclib as dparc
import dplib as dp
import measurementsettings, sessionmod
from experiment import Experiment

class MemoryCapacity(Experiment):
    """
    Memory capacity evaluation routine.
    DOES NOT NEED setMeasurement() to work.
    DOES NEED v_bias_vector passed to run().

    ex:
    mem=MemoryCapacity()
    mem.run(v_bias_vector = dp.generate_random_array(100,0,1))

    This experiment feeds a random generated waveform in input
    and evaluates the memory capacity of the network.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)
        self.sample_time = 0.5
        self.n_samples = 3000
        self.v_read = 0.05
        self.n_reps_avg = 1
        self.script = "MemoryCapacity"

    def setMeasurement(
        self,
        sample_time: float,
        n_samples: int,
        v_read: float,
        n_reps_avg: float,
    ):
        self.sample_time = sample_time
        self.n_samples = n_samples
        self.v_read = v_read
        self.n_reps_avg = n_reps_avg
        self.settings.vTimes = np.linspace(0, n_samples*sample_time, n_samples)

    def writeLogFile(self, f):
        """Writes Log file for Memory Capacity measurement routine"""
        f.write(f"DATE: {self.session.date}\n")
        f.write(f"LAB: {self.session.lab}\n")
        f.write(f"SAMPLE: {self.session.sample}\n")
        f.write(f"CELL: {self.session.cell}\n")
        f.write(f"EXPERIMENT: {self.name}, {self.script} script. \n\n")

        f.write("Experiment parameters:\n\n")
        f.write(f"Sample time: {self.sample_time} \n")
        f.write(f"Read voltage: {self.v_read} \n")
        f.write(f"Number of measurements per configuration: {self.n_reps_avg} \n")

        f.write("USED CHANNELS: \n\n")
        f.write(f"All channels: {self.settings.mask} \n")
        f.write("All two-channels combinations have been performed.")
        f.write("All other channels are set to floating each time.\n")

        f.flush()
        os.fsync(f.fileno())
        f.close()
        return

    def initializeFiles(self):
        self.session.num = (
            dp.findMaxNum(f"{self.session.savepath}/{self.session.sample}") + 1
        )
        self.session.dateUpdate()
        data_filename = f"{str(self.session.num).zfill(3)}_{self.session.lab}_{self.session.sample}_{self.session.cell}_{self.script}_{self.session.date}"
        data_file = dp.fileInit(
            f"{self.session.savepath}/{self.session.sample}",
            data_filename,
            self.headerInit(self.settings.mask),
        )

        log_file = dp.fileInit(
            f"{self.session.savepath}/{self.session.sample}",
            f"{data_filename}_log",
            "",
        )
        self.writeLogFile(log_file)
        return data_file, data_filename

    def run(self, v_bias_vector):
        """
        Performs Memory Capacity routine.
        """

        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()

        # ----------------------- routine -------------------------------------
        start_prog = time.time()
        progress_timer = time.time()
        data = {}
        dparc.setAllChannelsToFloat(self.arc)
        print(self.settings.vTimes)
        if len(v_bias_vector)>len(self.settings.vTimes):
            v_bias_vector = v_bias_vector[:len(self.settings.vTimes)]
            
        elif len(self.settings.vTimes)>len(v_bias_vector):
            self.settings.vTimes = self.settings.vTimes[:len(v_bias_vector)]
        print(f"bias len:{len(v_bias_vector)}, times len:{len(self.settings.vTimes)}")
        try:
            # cycle over the samples
            for sample_step, v_bias in enumerate(v_bias_vector):
                # set sleep until next sample for all iterations but the first
                if sample_step != 0:
                    waitfor = (
                        self.settings.vTimes[sample_step]
                        - time.time()
                        + start_prog
                        - 0.01
                    ) / self.settings.meas_iterations
                    if waitfor < 0:
                        waitfor = 0
                    time.sleep(waitfor)
                
                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                
                self.arc.config_channels(
                    biasedMask, base=None
                ).execute()  # set channels
                
                timestamp_sample, voltage_sample, current_sample = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
                # time.sleep(self.sample_time)
                if sample_step == 0:
                    timestamp = np.array([timestamp_sample])
                    voltage = np.array(voltage_sample)
                    current = np.array(current_sample)
                else:
                    timestamp = np.append(timestamp, timestamp_sample)
                    voltage = np.vstack((voltage, voltage_sample))
                    current = np.vstack((current, current_sample))
                # print the measurement progress at the end of the iteration
                # if at least 10 seconds have passed or if the measurement is completed

            dparc.setAllChannelsToFloat(self.arc)

            for sample in range(np.shape(voltage)[0]):
                data_row = dp.measureToStr(
                    timestamp[sample],
                    voltage[sample],
                    current[sample],
                    self.settings.mask,
                )
                dp.fileUpdate(data_file, data_row)

            data_file.close()
            return f"{self.session.savepath}/{self.session.sample}/{data_filename}.txt"
        except KeyboardInterrupt:
            dparc.setAllChannelsToFloat(self.arc)

            for sample in range(np.shape(voltage)[0]):
                data_row = dp.measureToStr(
                    timestamp[sample],
                    voltage[sample],
                    current[sample],
                    self.settings.mask,
                )
                dp.fileUpdate(data_file, data_row)
            data_file.close()
        return f"{self.session.savepath}/{self.session.sample}/{data_filename}.txt"