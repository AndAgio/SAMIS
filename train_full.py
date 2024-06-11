from src.utils import gather_settings
from src.bin import UniversalTrainer


def main():
    print('Gathering settings...')
    settings = gather_settings()

    assert settings.coreset_func == 'full'

    experiment_name = 'standard_{}_over_{}_with_{}_sched_{}'.format(settings.model, settings.dataset, settings.optimizer, settings.lr_sched)

    print('Setting up trainer...')
    trainer = UniversalTrainer(settings, experiment_name=experiment_name)

    if settings.resume:
        trainer.load_last_resume_ckpt()
    else:
        trainer.initialize_train()
    
    print('Running train...')
    trainer.train()
    print('Done!')


if __name__ == '__main__':
    main()