import ABC


class FeatureSelector(Object):
    '''
    Interface for feature selector classes

    Feature selector classes must extend this class to ensure a proper
    interface is used.

    Concrete implementations of this class must implement the
    static method select()

    E.x.
    class FeatureSelectorChris(FeatureSelector):
        ...
        @staticmethod
        def select():
            ...
            return numpy_array
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select(numpy_feature_array):
        '''
        Concrete implementations must implement this as a @staticmethod

        Implementations need to accept a raw numpy_feature_array
        and return a numpy_array that has features removed.
        '''
        pass