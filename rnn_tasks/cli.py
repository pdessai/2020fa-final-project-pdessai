from luigi import build

from .tasks import ByDecade, ByReview


def main(args=None):
    build([
        ByDecade(
            save_path='data/',
            save_glob='decade.parquet',
            save_flag=None
        ),
        ByReview(
            save_path='data/',
            save_glob='decade.parquet',
            save_flag=None
        )
        ], local_scheduler=True)