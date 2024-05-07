import unittest
from unittest.mock import Mock, patch

import weaviate

from database_utils import (
    Database,  # Replace 'your_module' with the actual name of your module
)


class TestDatabase(unittest.TestCase):

    @patch("database_utils.weaviate.connect_to_wcs")
    def test_singleton_behavior(self, connect_to_wcs_mock):
        connect_to_wcs_mock.return_value = Mock(name="WeaviateClient")

        instance1 = Database()
        instance2 = Database()

        self.assertIs(instance1, instance2)
        del instance1
        del instance2

    @patch("database_utils.os.getenv")
    @patch("database_utils.weaviate.connect_to_wcs")
    def test_database_initialization(self, connect_to_wcs_mock, os_getenv_mock):
        connect_to_wcs_mock.return_value = Mock(name="WeaviateClient")

        database = Database()

        connect_to_wcs_mock.assert_called_once_with(
            cluster_url=os_getenv_mock.return_value,
            auth_credentials=weaviate.auth.AuthApiKey(os_getenv_mock.return_value),
        )
        os_getenv_mock.assert_any_call("WCS_DEMO_URL")
        os_getenv_mock.assert_any_call("WCS_DEMO_RO_KEY")
        del database

    @patch("database_utils.WeaviateVectorStore")
    def test_set_db(self, weaviate_vector_store_mock):
        with patch("database_utils.weaviate.connect_to_wcs") as connect_to_wcs_mock:
            connect_to_wcs_mock.return_value = Mock(name="WeaviateClient")
            database = Database()
            database.set_db()

            weaviate_vector_store_mock.assert_called_once_with(
                client=database.client, index_name="MyIndex", text_key="text"
            )
            self.assertIsNotNone(database.db)
            del database

    @patch("database_utils.Database.set_db")
    def test_get_db(self, set_db_mock):
        database = Database()
        self.assertIsNone(database.db)  # before calling get_db()

        result = database.get_db()

        set_db_mock.assert_called_once()
        self.assertEqual(
            result, database.db
        )  # `get_db` initializes `db.db` if it's None
        del database
