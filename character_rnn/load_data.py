from django.core.management import BaseCommand


class Command(BaseCommand):
    help = "Load review facts"

    def add_arguments(self, parser):
        parser.add_argument("-f", "--full", action="store_true")

    def handle(self, *args, **options):
        """Populates the DB with review aggregations"""
        # Load the data
        # Calculate daily aggregations
        # Store results into FactReview

        df_records = df.to_dict('records')

        model_instances = [MyModel(
            field_1=record['field_1'],
            field_2=record['field_2'],
        ) for record in df_records]

        MyModel.objects.bulk_create(model_instances)