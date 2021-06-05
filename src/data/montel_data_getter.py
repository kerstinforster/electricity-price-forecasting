""" This class implements a data getter that downloads the montel electricity
stock market price data"""

import requests
import json

from data_getter import DataGetter


class MontelDataGetter(DataGetter):
    """
    This class is responsible for downloading the Montel dataset(s)
    """
    def __init__(self):
        """
        Constructor for the Montel Data Getter
        """
        super().__init__("montel")
        self.token = "Zeo4XP4SswsOwnradSkZQ6Xh16eEi10UNtTE1Q3ek4lS2Xv0Nx40RKs" \
                     "KqxryzUCwAV6fzmt2erLFigT9iNy_IYcq85A7jO-kE7mRun8Dpk6BH6" \
                     "xc0mMVzzyog8ZnK3Jk3X_8drbMbGqMUeFGc7ul06CERBN_QZ4ySQdOR" \
                     "1EAYOLzVyHkae7d3KBdxLazP3QvYohPhIYGScKmNipVkhOpTdQwROce" \
                     "ZS54CGCw8QSWm8wv5vnCNXYxAl7fxd4nEZEaZhHqOgKMnw2MgjtHYJ2" \
                     "jkhfnsOQcbP52zp-6wI6-YyXWoiOphouS7w10Le0kacBQfDXVzMjoGS" \
                     "FIdPCRxB4qTrXT6n14OO2VwdrXVO4wxbYN65fUYWntRE02d820pw4fE" \
                     "rykZaUcQFCTEqWC9XHU_FWlSC2yZAGNrMFBrh5EiZ4"
        self.token_check()

    def token_check(self) -> None:
        """
        This function checks whether the token is still valid
        :raise PermissionError: if Bearer Token invalid
        """
        response = requests.get(
            "https://api.montelnews.com/spot/getmetadata",
            headers={"Authorization": f"Bearer {self.token}"})
        if response.status_code != 200:
            raise PermissionError(f"The MontelBearer Token seems to be invalid,"
                                  f" status {response.status_code}, \nresponse:"
                                  f" {response.text}")

    def show_available_datasets(self) -> None:
        """
        This function can be used to print all available datasets provided by
        Montel.
        """
        response = requests.get(
            "https://api.montelnews.com/spot/getmetadata",
            headers={"Authorization": f"Bearer {self.token}"})
        results = json.loads(response.text)["Elements"]
        for result in results:
            # Other fields are: 'SpotKey', 'SpotName', 'SourceName', 'PeakType',
            # 'Country', 'DefaultCurrency', 'AvailableCurrencies'
            print(result["SpotName"])

    def get_data(self) -> bool:
        """
        This function downloads the datasets that shall be used for training.
        It stores them in the folder data/montel
        :return: bool True if successful, otherwise False
        """
        return False


if __name__ == "__main__":
    dg = MontelDataGetter()
    dg.show_available_datasets()
