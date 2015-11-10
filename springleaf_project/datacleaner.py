import ABC


class DataCleaner(Object):
    '''
    Interface for data cleaner classes

    Data cleaner classes must extend this class to ensure a proper
    interface is used.

    Concrete implementations of this class must implement the
    static method clean()

    E.x.
    class DataCleanerChris(DataCleaner):
        ...
        @staticmethod
        def preprocess():
            ...
            return numpy_array
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def clean(numpy_feature_array):
        '''
        Concrete implementations must implement this as a @staticmethod

        Implementations need to accept a raw numpy_feature_array
        and return a numpy_array that has been cleaned.
        '''
        pass