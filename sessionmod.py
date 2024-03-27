# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:28:38 2023

@author: Davide Pilati
"""
from datetime import date
import dplib as dp
import tomli


class Session:
    def __init__(
        self,
    ):
        settings = self.loadToml("config/save_config.toml","rb")
        self.savepath = settings["savepath"]
        self.lab = settings["lab"]
        self.sample = settings["sample"]
        self.cell = settings["cell"]
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
    
    def loadToml(filename) -> dict:
        """Load toml data from file"""

        with open(filename, "rb") as f:
            toml_data: dict = tomli.load(f)
            return toml_data

