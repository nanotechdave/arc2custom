# Form implementation generated from reading ui file 'C:\Users\mcfab\AppData\Local\Temp\pip-req-build-8ggypqms\uis\about.ui'
#
# Created by: PyQt6 UI code generator 6.4.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_AboutDialog(object):
    def setupUi(self, AboutDialog):
        AboutDialog.setObjectName("AboutDialog")
        AboutDialog.resize(513, 318)
        AboutDialog.setStyleSheet("")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(AboutDialog)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.logoLabel = QtWidgets.QLabel(AboutDialog)
        self.logoLabel.setStyleSheet("background: #002060;")
        self.logoLabel.setText("")
        self.logoLabel.setObjectName("logoLabel")
        self.verticalLayout_2.addWidget(self.logoLabel)
        self.holderWidget = QtWidgets.QWidget(AboutDialog)
        self.holderWidget.setStyleSheet("QWidget#holderWidget { background-color: white; }")
        self.holderWidget.setObjectName("holderWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.holderWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.holderWidget)
        self.label_2.setStyleSheet("font-weight: bold;")
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.copyrightYearLabel = QtWidgets.QLabel(self.holderWidget)
        self.copyrightYearLabel.setObjectName("copyrightYearLabel")
        self.horizontalLayout_2.addWidget(self.copyrightYearLabel)
        self.label_4 = QtWidgets.QLabel(self.holderWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.label_5 = QtWidgets.QLabel(self.holderWidget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.label_3 = QtWidgets.QLabel(self.holderWidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_6 = QtWidgets.QLabel(self.holderWidget)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        spacerItem1 = QtWidgets.QSpacerItem(20, 16, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.label_7 = QtWidgets.QLabel(self.holderWidget)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.holderWidget)
        self.label.setStyleSheet("font-weight: bold;")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.versionLabel = QtWidgets.QLabel(self.holderWidget)
        self.versionLabel.setMinimumSize(QtCore.QSize(5, 0))
        self.versionLabel.setText("")
        self.versionLabel.setObjectName("versionLabel")
        self.horizontalLayout.addWidget(self.versionLabel)
        self.label_10 = QtWidgets.QLabel(self.holderWidget)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout.addWidget(self.label_10)
        self.label_8 = QtWidgets.QLabel(self.holderWidget)
        self.label_8.setStyleSheet("font-weight: bold;")
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        self.pythonVersionLabel = QtWidgets.QLabel(self.holderWidget)
        self.pythonVersionLabel.setMinimumSize(QtCore.QSize(5, 0))
        self.pythonVersionLabel.setText("")
        self.pythonVersionLabel.setObjectName("pythonVersionLabel")
        self.horizontalLayout.addWidget(self.pythonVersionLabel)
        self.label_11 = QtWidgets.QLabel(self.holderWidget)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout.addWidget(self.label_11)
        self.label_9 = QtWidgets.QLabel(self.holderWidget)
        self.label_9.setStyleSheet("font-weight: bold;")
        self.label_9.setObjectName("label_9")
        self.horizontalLayout.addWidget(self.label_9)
        self.qtVersionLabel = QtWidgets.QLabel(self.holderWidget)
        self.qtVersionLabel.setMinimumSize(QtCore.QSize(5, 0))
        self.qtVersionLabel.setText("")
        self.qtVersionLabel.setObjectName("qtVersionLabel")
        self.horizontalLayout.addWidget(self.qtVersionLabel)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addWidget(self.holderWidget)
        self.line = QtWidgets.QFrame(AboutDialog)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.buttonBoxWidget = QtWidgets.QWidget(AboutDialog)
        self.buttonBoxWidget.setStyleSheet("")
        self.buttonBoxWidget.setObjectName("buttonBoxWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.buttonBoxWidget)
        self.verticalLayout_3.setContentsMargins(-1, 9, -1, 9)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.buttonBoxWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_3.addWidget(self.buttonBox)
        self.verticalLayout_2.addWidget(self.buttonBoxWidget)

        self.retranslateUi(AboutDialog)
        self.buttonBox.accepted.connect(AboutDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(AboutDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(AboutDialog)

    def retranslateUi(self, AboutDialog):
        _translate = QtCore.QCoreApplication.translate
        AboutDialog.setWindowTitle(_translate("AboutDialog", "About ArC2 Control Panel"))
        self.label_2.setText(_translate("AboutDialog", "Graphical Interface for the ArC TWO Characterisation Platform"))
        self.copyrightYearLabel.setText(_translate("AboutDialog", "©2022"))
        self.label_4.setText(_translate("AboutDialog", " – ArC Instruments Ltd."))
        self.label_5.setText(_translate("AboutDialog", "<html>\n"
"<head/>\n"
" <body>\n"
"  <p>\n"
"    Home Page: <a href=\"http://arc-instruments.co.uk\"><span style=\" text-decoration: underline; color:#0000ff;\">http://arc-instruments.co.uk</span></a><br/>\n"
"   GitHub: <a href=\"https://github.com/arc-instruments\"><span style=\" text-decoration: underline; color:#0000ff;\">https://github.com/arc-instruments</span>\n"
"  </p>\n"
" </body>\n"
"</html>"))
        self.label_3.setText(_translate("AboutDialog", "<html><head/><body><p>ArC2 Control Panel is developed by:<br/>Spyros Stathopoulos &lt;<a href=\"mailto:spyros@arc-instruments.co.uk\"><span style=\" text-decoration: underline; color:#0000ff;\">spyros@arc-instruments.co.uk</span></a>&gt;</p></body></html>"))
        self.label_6.setText(_translate("AboutDialog", "<html><head/><body><p>ArC2 Control Panel is licensed under <a href=\"https://www.gnu.org/licenses/lgpl-3.0.en.html\"><span style=\" text-decoration: underline; color:#0000ff;\">LGPL-3.0</span></a><br/>This software uses <a href=\"https://github.com/arc-instruments/libarc2\"><span style=\" text-decoration: underline; color:#0000ff;\">libarc2</span></a>, licensed under <a href=\"https://www.mozilla.org/en-US/MPL/2.0/\"><span style=\" text-decoration: underline; color:#0000ff;\">MPL-2.0</span></a><br/>This program comes with no warranty; see the license agreement for details</p></body></html>"))
        self.label_7.setText(_translate("AboutDialog", "<html><head/><body><p>Please report bugs to <a href=\"https://github.com/arc-instruments/arc2control/issues\"><span style=\" text-decoration: underline; color:#0000ff;\">https://github.com/arc-instruments/arc2control/issues</span></a></p></body></html>"))
        self.label.setText(_translate("AboutDialog", "ArC2Control Version:"))
        self.label_10.setText(_translate("AboutDialog", "–"))
        self.label_8.setText(_translate("AboutDialog", "Python:"))
        self.label_11.setText(_translate("AboutDialog", "–"))
        self.label_9.setText(_translate("AboutDialog", "Qt:"))
