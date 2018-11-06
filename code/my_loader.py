

class XrayLoader():
    def __init__(self, path):
        self.size = 0
        self.path = path


    def getNextBatch(self):
        data = []
        labels = []
        self.size += 1
        return (data, labels)