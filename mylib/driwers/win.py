from driwers import thorlabs_apt as apt
from driwers.TLPMall.TLPM import TLPM
from ctypes import c_uint32, byref, create_string_buffer, c_bool, c_int, c_double  # , c_void
import serial
import time

class motor(apt.Motor):
    def __init__(self, motor_id):
        self.__init__(motor_id)
        self.backlash_distance = 0.0
        self.set_hardware_limit_switches(2, 1)
        self.set_move_home_parameters(2, 1, 1.0, 0.5)

    def __del__(self):
        self.disable()

    def list_available_devices():
        apt.list_available_devices()


class powerMeter(TLPM):
    def __init__(self):
        TLPM.__init__(self)
        self.Connect()

    def __del__(self):
        self.Close()

    def Connect(self):
        deviceName = self.FindSingleDevice()

        if (deviceName == -1):
            print("Unable to Connect\n")
            return -1

        self.open(deviceName, c_bool(True), c_bool(True))
        self.isConnected = True
        return self.isConnected

    def FindSingleDevice(self):
        deviceCount = c_uint32()
        self.findRsrc(byref(deviceCount))

        if (deviceCount.value == 0):
            print("Device Not Found\n")
            return -1

        deviceName = create_string_buffer(1024)
        self.getRsrcName(c_int(0), deviceName)
        return deviceName.value

    def Read(self):
        power = c_double()
        self.measPower(byref(power))
        return power.value

    def Close(self):
        if (self.isConnected == False):
            print("Device not Connected\n")
            return -1
        self.close()
        self.isConnected = False
        return 0


class tensionGauge():
    def __init__(self):
        self.Connect()

    def Connect(self):
        self.port = serial.Serial(port="COM3", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
        time.sleep(1)  # действительно нужно
        self.port.write(1)
        time.sleep(1)
        self.isConnected = True
        return 0

    def Close(self):
        self.port.close()

    def read(self):
        self.port.write(1)
        string = ''
        i = 0
        while len(string) < 3:
            t0 = time.time()
            while self.simFlag == False and self.port.in_waiting == 0:
                if time.time() - t0 > 1:
                    print('tg read woiting problem')
                    return 'problem'
            else:
                string = self.port.readline()
            if len(string) < 3:
                i += 1
                if i > 10:
                    print('tg read problem')
                    return 'problem'
        # print(string)
        weight = float(string[0:-2])
        return weight
