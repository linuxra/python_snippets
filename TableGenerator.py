import configparser


class TableGenerator:
    """
    TableGenerator is a utility class that creates table objects based on a given
    configuration. This makes it easy to generate different types of tables with
    various styles and settings.
    """
    def __init__(self, config: configparser.ConfigParser):
        """
        Initializes a TableGenerator instance.

        :param config: A ConfigParser object containing the table settings.
        """
        self.config = config

    def generate_table(self, func_id: str, df: pd.DataFrame, from_date: str = None, to_date: str = None) -> Table:
        """
        Generate a table object based on the provided parameters and the
        settings from the configuration file.

        :param func_id: The identifier for the function. This is used to find
                        the corresponding settings in the configuration file.
        :param df: A pandas DataFrame to use as the data for the table.
        :param from_date: The start date as a string. This will be used to format the table title.
        :param to_date: The end date as a string. This will be used to format the table title.
        :return: A Table object configured according to the settings in the configuration file.
        """
        try:
            class_name = self.config.get(func_id, "class")
            table_class = globals()[class_name]
            table_title_template = self.config.get(func_id, "title")
            table_title = table_title_template.format(from_date, to_date)
            scale = self.config.get(func_id, "scale")
            annotations = self.config.get(func_id, "annotations")
            first_columns_to_color = self.config.get(func_id, "first_columns_to_color")

            table = table_class(df, table_title)
            table.scale = scale
            table.first_columns_to_color = first_columns_to_color
            table.add_annotations(annotations)

            return table

        except KeyError as e:
            print(f"Error: Invalid function identifier '{func_id}'.")
            return None

    def save_table_as_report(self, table: Table, func_id: str):
        """
        Saves a table to a file. The file name is taken from the configuration file.

        :param table: The Table object to save.
        :param func_id: The identifier for the function. This is used to find
                        the corresponding settings in the configuration file.
        """
        try:
            report_name = self.config.get(func_id, "report_name")
            table.save(report_name)
            print(f"Table saved as {report_name}.")
        except Exception as e:
            print(f"An error occurred while saving the table: {e}")

    def show_table(self, table: Table):
        """
        Displays a table.

        :param table: The Table object to display.
        """
        table.show()
