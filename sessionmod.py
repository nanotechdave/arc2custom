# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:28:38 2023

@author: Davide Pilati
"""

from pathlib import Path
from datetime import date
from arc2custom import dplib as dp
import tomli


class Session:
    def __init__(
        self,
    ):
        self.parentpath = Path(__file__).resolve().parent
        self.settings = self.loadToml(filename =f"{self.parentpath}/config/save_config.toml")
        self.savepath = self.settings["savepath"]
        self.lab = self.settings["lab"]
        self.sample = self.settings["sample"]
        self.cell = self.settings["cell"]
        self.date = date.today().strftime("%Y_%m_%d")
        print(self.savepath)
        dp.ensureDirectoryExists(f"{self.savepath}/{self.sample}")
        self.num = dp.findMaxNum(f"{self.savepath}/{self.sample}")
        print(self.savepath)

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
    
    def loadToml(self, filename) -> dict:
        """Load toml data from file"""

        with open(filename, "rb") as f:
            toml_data: dict = tomli.load(f)
            return toml_data

