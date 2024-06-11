from src.utils import gather_settings
from src.bin import UniversalTrainer


def main():
    print('Gathering settings...')
    settings = gather_settings()

    assert settings.coreset_func in ["craig", "glister", "gradmatch", "forget", "grand", "el2n", "infobatch", "graph_cut", "random",
                                    "min_mem", "min_forg", "min_etl", "min_eps", "min_flat", "min_sam-sgd-loss", "min_sam-sgd-prob"]

    experiment_name = '{}_static_coreset_{}_{}_{}_over_{}'.format(settings.coreset_func, settings.coreset_mode, settings.coreset_fraction, settings.model, settings.dataset)

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