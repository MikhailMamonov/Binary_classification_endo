config = {
    'vgg16': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 56
        },
        'fit_generator': {
            'epochs': 5
        }
    },
    'skin_rec': {
        'flow_generator': {
            'target_size': (141, 141),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 3
        }
    },
     'lung_rec': {
        'flow_generator': {
            'target_size': (141, 141),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'alex_net': {
        'flow_generator': {
            'target_size': (141, 141),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'vgg19': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'resnet50': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'densenet': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'incresnet': {
        'flow_generator': {
            'target_size': (299, 299),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'incv3': {
        'flow_generator': {
            'target_size': (141, 141),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'xcept': {
        'flow_generator': {
            'target_size': (299, 299),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 7
        }
    },
    'nasnet': {
        'flow_generator': {
            'target_size': (331, 331),
            'batch_size': 64
        },
        'fit_generator': {
            'epochs': 15
        }
    }
}