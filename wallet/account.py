class Account:

    def __init__(self):
        self.mnemonic = None
        self.privateKey = None
        self.address = None

    def __int__(self, address, privateKey, mnemonic):
        self.address = address
        self.privateKey = privateKey
        self.mnemonic = mnemonic

    def setAddress(self, address):
        self.address = address

    def setPrivateKey(self, privateKey):
        self.privateKey = privateKey

    def setMnemonic(self, mnemonic):
        self.mnemonic = mnemonic
