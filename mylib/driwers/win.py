from driwers import thorlabs_apt as apt
from driwers.TLPMall.TLPM import TLPM
from ctypes import c_uint32, byref, create_string_buffer, c_bool, c_int, c_double  # , c_void
import serial
from driwers.newport import Controller
import time


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class motor(apt.Motor):
    def __init__(self, motor_id):
        apt.Motor.__init__(self, motor_id)
        self.backlash_distance = 0.0
        self.set_hardware_limit_switches(2, 1)
        self.set_move_home_parameters(2, 1, 1.0, 0.5)

    def __del__(self):
        self.disable()

    def list_available_devices():
        return apt.list_available_devices()


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

    def read(self):
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
        while len(string) < 3 or not isfloat(string):
            t0 = time.time()
            while self.port.in_waiting == 0:
                if time.time() - t0 > 2:
                    print('tg read waiting problem')
                    return 'problem'
            string = self.port.readline()
            if len(string) < 3:
                i += 1
                if i > 10:
                    print('tg read problem')
                    return 'problem'
            # print(string)
        weight = float(string[0:-2])
        return weight


class tikalka_base():
    _controller = Controller(idProduct=0x4000, idVendor=0x104d)
    ides = {'x': 3, 'y': 2, 'z': 1}

    def __init__(self, name):
        self.id = tikalka_base.ides[name]
        self.coord = 0

    def IsInMotion(self):
        motor_done_cmd = '{}MD?'.format(self.id)
        resp = tikalka_base._controller.command(motor_done_cmd)
        return not int(resp[2])  # True if motor in motion

    def move(self, value):
        while self.IsInMotion():
            pass
        move_motor_cmd = '{}PR{}'.format(self.id, int(value))
        # print(move_motor_cmd)
        tikalka_base._controller.command(move_motor_cmd)
        self.coord += int(value)

    def move_to(self, value):
        self.move(value - self.coord)

    # def move_absolute(self, motor_id, value):
    #     move_motor_cmd = '{}PA{}'.format(motor_id, value)
    #     self._controller.command(move_motor_cmd)
    #
    # def get_home_position(self, motor_id):
    #     return int(self._controller.command('{}DH?'.format(motor_id))[2:])
    #
    # def get_position(self, motor_id):
    #     return int(self._controller.command('{}TP?'.format(motor_id))[2:])
    #
    # def set_home_position(self, motor_id, value):
    #     self._controller.command('{}DH{}'.format(motor_id, value))
    #
    # def get_target(self, motor_id):
    #     return int(self._controller.command('{}PA?'.format(motor_id))[2:])
