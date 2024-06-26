# Form implementation generated from reading ui file 'C:\Users\mcfab\AppData\Local\Temp\pip-req-build-8ggypqms\arc2control\modules\curvetracer\uis\curvetracer.ui'
#
# Created by: PyQt6 UI code generator 6.4.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_CurveTracerWidget(object):
    def setupUi(self, CurveTracerWidget):
        CurveTracerWidget.setObjectName("CurveTracerWidget")
        CurveTracerWidget.resize(334, 206)
        self.gridLayout = QtWidgets.QGridLayout(CurveTracerWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(0, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem, 6, 4, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout.addItem(spacerItem1, 11, 0, 1, 1)
        self.rampVNegMaxSpinBox = QtWidgets.QDoubleSpinBox(CurveTracerWidget)
        self.rampVNegMaxSpinBox.setMaximum(10.0)
        self.rampVNegMaxSpinBox.setProperty("value", 1.0)
        self.rampVNegMaxSpinBox.setObjectName("rampVNegMaxSpinBox")
        self.gridLayout.addWidget(self.rampVNegMaxSpinBox, 3, 1, 1, 1)
        self.rampPulsesSpinBox = QtWidgets.QSpinBox(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rampPulsesSpinBox.sizePolicy().hasHeightForWidth())
        self.rampPulsesSpinBox.setSizePolicy(sizePolicy)
        self.rampPulsesSpinBox.setMinimum(1)
        self.rampPulsesSpinBox.setMaximum(1000)
        self.rampPulsesSpinBox.setProperty("value", 1)
        self.rampPulsesSpinBox.setObjectName("rampPulsesSpinBox")
        self.gridLayout.addWidget(self.rampPulsesSpinBox, 1, 3, 1, 1)
        self.biasTypeComboBox = QtWidgets.QComboBox(CurveTracerWidget)
        self.biasTypeComboBox.setObjectName("biasTypeComboBox")
        self.gridLayout.addWidget(self.biasTypeComboBox, 3, 3, 1, 1)
        self.label_5 = QtWidgets.QLabel(CurveTracerWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 2, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.rampSelectedButton = QtWidgets.QPushButton(CurveTracerWidget)
        self.rampSelectedButton.setObjectName("rampSelectedButton")
        self.horizontalLayout.addWidget(self.rampSelectedButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.gridLayout.addLayout(self.horizontalLayout, 12, 0, 1, 5)
        self.ivTypeComboBox = QtWidgets.QComboBox(CurveTracerWidget)
        self.ivTypeComboBox.setObjectName("ivTypeComboBox")
        self.gridLayout.addWidget(self.ivTypeComboBox, 4, 3, 1, 1)
        self.rampVStartSpinBox = QtWidgets.QDoubleSpinBox(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rampVStartSpinBox.sizePolicy().hasHeightForWidth())
        self.rampVStartSpinBox.setSizePolicy(sizePolicy)
        self.rampVStartSpinBox.setMinimum(-10.0)
        self.rampVStartSpinBox.setMaximum(10.0)
        self.rampVStartSpinBox.setSingleStep(0.1)
        self.rampVStartSpinBox.setObjectName("rampVStartSpinBox")
        self.gridLayout.addWidget(self.rampVStartSpinBox, 0, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(CurveTracerWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 2, 1, 1)
        self.label = QtWidgets.QLabel(CurveTracerWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 6, 0, 1, 1)
        self.readAtComboBox = QtWidgets.QComboBox(CurveTracerWidget)
        self.readAtComboBox.setObjectName("readAtComboBox")
        self.gridLayout.addWidget(self.readAtComboBox, 5, 3, 1, 1)
        self.rampPwDurationWidget = DurationWidget(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rampPwDurationWidget.sizePolicy().hasHeightForWidth())
        self.rampPwDurationWidget.setSizePolicy(sizePolicy)
        self.rampPwDurationWidget.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        self.rampPwDurationWidget.setObjectName("rampPwDurationWidget")
        self.gridLayout.addWidget(self.rampPwDurationWidget, 5, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 5, 0, 1, 1)
        self.rampCyclesSpinBox = QtWidgets.QSpinBox(CurveTracerWidget)
        self.rampCyclesSpinBox.setMinimum(1)
        self.rampCyclesSpinBox.setMaximum(1000)
        self.rampCyclesSpinBox.setObjectName("rampCyclesSpinBox")
        self.gridLayout.addWidget(self.rampCyclesSpinBox, 0, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(CurveTracerWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 2, 1, 1)
        self.rampVPosMaxSpinBox = QtWidgets.QDoubleSpinBox(CurveTracerWidget)
        self.rampVPosMaxSpinBox.setMinimum(0.0)
        self.rampVPosMaxSpinBox.setMaximum(10.0)
        self.rampVPosMaxSpinBox.setProperty("value", 1.0)
        self.rampVPosMaxSpinBox.setObjectName("rampVPosMaxSpinBox")
        self.gridLayout.addWidget(self.rampVPosMaxSpinBox, 1, 1, 1, 1)
        self.rampInterDurationWidget = DurationWidget(CurveTracerWidget)
        self.rampInterDurationWidget.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        self.rampInterDurationWidget.setObjectName("rampInterDurationWidget")
        self.gridLayout.addWidget(self.rampInterDurationWidget, 6, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 4, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 0, 0, 1, 1)
        self.rampVStepSpinBox = QtWidgets.QDoubleSpinBox(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rampVStepSpinBox.sizePolicy().hasHeightForWidth())
        self.rampVStepSpinBox.setSizePolicy(sizePolicy)
        self.rampVStepSpinBox.setMinimum(0.0)
        self.rampVStepSpinBox.setMaximum(10.0)
        self.rampVStepSpinBox.setSingleStep(0.1)
        self.rampVStepSpinBox.setProperty("value", 0.1)
        self.rampVStepSpinBox.setObjectName("rampVStepSpinBox")
        self.gridLayout.addWidget(self.rampVStepSpinBox, 4, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(CurveTracerWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(CurveTracerWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(CurveTracerWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.label_5.setBuddy(self.ivTypeComboBox)
        self.label_6.setBuddy(self.readAtComboBox)
        self.label.setBuddy(self.rampVPosMaxSpinBox)
        self.label_13.setBuddy(self.rampInterDurationWidget)
        self.label_12.setBuddy(self.rampPwDurationWidget)
        self.label_4.setBuddy(self.biasTypeComboBox)
        self.label_10.setBuddy(self.rampVStepSpinBox)
        self.label_9.setBuddy(self.rampVStartSpinBox)
        self.label_3.setBuddy(self.rampCyclesSpinBox)
        self.label_14.setBuddy(self.rampPulsesSpinBox)
        self.label_2.setBuddy(self.rampVNegMaxSpinBox)

        self.retranslateUi(CurveTracerWidget)
        QtCore.QMetaObject.connectSlotsByName(CurveTracerWidget)
        CurveTracerWidget.setTabOrder(self.rampVStartSpinBox, self.rampVPosMaxSpinBox)
        CurveTracerWidget.setTabOrder(self.rampVPosMaxSpinBox, self.rampVNegMaxSpinBox)
        CurveTracerWidget.setTabOrder(self.rampVNegMaxSpinBox, self.rampVStepSpinBox)
        CurveTracerWidget.setTabOrder(self.rampVStepSpinBox, self.rampPwDurationWidget)
        CurveTracerWidget.setTabOrder(self.rampPwDurationWidget, self.rampInterDurationWidget)
        CurveTracerWidget.setTabOrder(self.rampInterDurationWidget, self.rampCyclesSpinBox)
        CurveTracerWidget.setTabOrder(self.rampCyclesSpinBox, self.rampPulsesSpinBox)
        CurveTracerWidget.setTabOrder(self.rampPulsesSpinBox, self.biasTypeComboBox)
        CurveTracerWidget.setTabOrder(self.biasTypeComboBox, self.ivTypeComboBox)
        CurveTracerWidget.setTabOrder(self.ivTypeComboBox, self.readAtComboBox)
        CurveTracerWidget.setTabOrder(self.readAtComboBox, self.rampSelectedButton)

    def retranslateUi(self, CurveTracerWidget):
        _translate = QtCore.QCoreApplication.translate
        CurveTracerWidget.setWindowTitle(_translate("CurveTracerWidget", "Form"))
        self.rampVNegMaxSpinBox.setSuffix(_translate("CurveTracerWidget", " V"))
        self.label_5.setText(_translate("CurveTracerWidget", "IV direction"))
        self.rampSelectedButton.setText(_translate("CurveTracerWidget", "Apply to Selected"))
        self.rampVStartSpinBox.setSuffix(_translate("CurveTracerWidget", " V"))
        self.label_6.setText(_translate("CurveTracerWidget", "Read At"))
        self.label.setText(_translate("CurveTracerWidget", "Positive Vmax"))
        self.label_13.setText(_translate("CurveTracerWidget", "Inter"))
        self.label_12.setText(_translate("CurveTracerWidget", "PW"))
        self.label_4.setText(_translate("CurveTracerWidget", "Bias Type"))
        self.rampVPosMaxSpinBox.setSuffix(_translate("CurveTracerWidget", " V"))
        self.label_10.setText(_translate("CurveTracerWidget", "Voltage step"))
        self.label_9.setText(_translate("CurveTracerWidget", "Initial Voltage"))
        self.rampVStepSpinBox.setSuffix(_translate("CurveTracerWidget", " V"))
        self.label_3.setText(_translate("CurveTracerWidget", "Cycles"))
        self.label_14.setText(_translate("CurveTracerWidget", "Pulses"))
        self.label_2.setText(_translate("CurveTracerWidget", "Negative Vmax"))
from arc2control.widgets.duration_widget import DurationWidget
