from django.shortcuts import render
from django.http import JsonResponse

datasets = [{'name': 'Aahaber', 'code': 'aahaber'},
            {'name': 'Hurriyet', 'code': 'hurriyet'},
            {'name': 'Milliyet', 'code': 'milliyet'},
            {'name': '17K Tweet', 'code': '17k'},
            {'name': '3K Tweet', 'code': '3k'},
            {'name': 'MiniNews', 'code': 'mininews'},
            ]
algorithms = [{'name': 'Multilayer Perceptron', 'code': 'mlp'},
              {'name': 'Perceptron (One Layer)', 'code': 'ol'},
              {'name': 'Convolutional Neural Network', 'code': 'cnn'},
              {'name': 'Recurrent Neural Network', 'code': 'rnn'},
              {'name': 'Long Short-Term Memory', 'code': 'lstm'},
              {'name': 'FastText', 'code': 'ft'}, ]


def home(request):
    context = {
        'datasets': datasets,
        'algorithms': algorithms
    }
    return render(request, 'gui/home.html', context)


def about(request):
    return render(request, 'gui/about.html', {'t': 'test'})


def evaluate(request):

    dataset = request.POST.get('dataset')
    algorithm = request.POST.get('algorithm')
    test_size = request.POST.get('test_size')

    TEST_SIZE = 0
    if test_size == 't20':
        TEST_SIZE = 0.2
    elif test_size == 't50':
        TEST_SIZE = 0.5
    else:
        TEST_SIZE = 0.7

    Dataset = ''
    if dataset == 'aahaber':
        from .lib.Library.Aahaber import Aahaber
        Dataset = Aahaber(False, True)
    elif dataset == 'hurriyet':
        from .lib.Library.Hurriyet import Hurriyet
        Dataset = Hurriyet(False, True)
    elif dataset == 'milliyet':
        from .lib.Library.Milliyet import Milliyet
        Dataset = Milliyet(False, True)
    elif dataset == '17k':
        from .lib.Library.Tweet17K import Tweet17K
        Dataset = Tweet17K(True, True)
    elif dataset == '3k':
        from .lib.Library.Tweet3K import Tweet3K
        Dataset = Tweet3K(True, True)
    else:
        from .lib.Library.MiniNews import MiniNews
        Dataset = MiniNews(False, True)

    Processor = ''
    if dataset == 'mininews':
        from .lib.Library.EnglishProcessor import EnglishProcessor
        Processor = EnglishProcessor(Dataset)
    else:
        from .lib.Library.TurkishProcessor import TurkishProcessor
        Processor = TurkishProcessor(Dataset)

    Model = ''
    if algorithm == 'mlp':
        from .lib.Library.MlpModel import MlpModel
        Model = MlpModel(Processor, Dataset, TEST_SIZE)
    elif algorithm == 'ol':
        from .lib.Library.OneLayerModel import OneLayerModel
        Model = OneLayerModel(Processor, Dataset, TEST_SIZE)
    elif algorithm == 'cnn':
        from .lib.Library.CnnModel import CnnModel
        Model = CnnModel(Processor, Dataset, TEST_SIZE)
    elif algorithm == 'rnn':
        from .lib.Library.RnnModel import RnnModel
        Model = RnnModel(Processor, Dataset, TEST_SIZE)
    elif algorithm == 'lstm':
        from .lib.Library.LstmModel import LstmModel
        Model = LstmModel(Processor, Dataset, TEST_SIZE)
    else:
        from .lib.Library.FastTextModel import FastTextModel
        Model = FastTextModel(Processor, Dataset, TEST_SIZE)

    history = Model.evaluate()
    epochs = []
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    for i in range(len(history.history['val_loss'])):
        epochs.append(i+1)
        train_acc.append(history.history['acc'][i])
        train_loss.append(history.history['loss'][i])
        test_acc.append(history.history['val_acc'][i])
        test_loss.append(history.history['val_loss'][i])

    model_name = str(Model)
    dataset_name = str(Dataset)

    context = {
        'datasets': datasets,
        'algorithms': algorithms,
        'epochs': epochs,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'model_name': model_name,
        'dataset_name': dataset_name
    }

    return render(request, 'gui/home.html', context)
