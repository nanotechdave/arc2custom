# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:28:38 2023

@author: Davide Pilati
"""
from datetime import date
import dplib as dp


class Session:
    def __init__(
        self,
        savepath: str = "C:/Users/mcfab/Desktop/Measurements",
        lab: str = "INRiMC110",
        sample: str = "NWN_Pad44D",
        cell: str = "13N3-17S3-14W3-16E3",
    ):
        self.savepath = savepath
        self.lab = lab
        self.sample = sample
        self.cell = cell
        self.date = date.today().strftime("%Y_%m_%d")
        dp.ensureDirectoryExists(f"{self.savepath}/{self.sample}")
        self.num = dp.findMaxNum(f"{self.savepath}/{self.sample}")

    def setSave(self, savepath: str, lab: str, sample: str, cell: str):
        self.savepath = savepath
        self.lab = lab
        self.sample = sample
        self.cell = cell
        dp.ensureDirectoryExists(f"{self.savepath}/{self.sample}")
        self.num = dp.findMaxNum(f"{savepath}/{sample}") + 1
        return

    def dateUpdate(self):
        self.date = date.today().strftime("%Y_%m_%d")
        return
