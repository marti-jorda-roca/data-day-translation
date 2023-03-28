"""This module contains the models used to make predictions."""
import abc
import json
from dataclasses import dataclass

import boto3
from easynmt import EasyNMT


@dataclass(frozen=True, kw_only=True)
class Language:
    """Enum for languages."""
    spanish = "Spanish"
    italian = "Italian"
    portugees = "Portugees"
    catalan = "Catalan"
    galician = "Galician"


class TranslateModel(metaclass=abc.ABCMeta):
    """Abstract class for the models used to make predictions."""

    def __init__(self, endpoint_name: str):
        """Class constructor.

        Args:
            endpoint_name (str): The name of the endpoint to use.
        """
        self.endpoint_name = endpoint_name
        self.client = boto3.client("sagemaker-runtime")

    @abc.abstractmethod
    def predict(self, text: str, source_lang: str, target_lang: str) -> str:
        """Abstract method to make predictions.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language.
            target_lang (str): The target language.

        Returns:
            str: The translated text.
        """
        raise NotImplementedError()

    def _invoke_sagemaker_endpoint(self, body: str) -> dict:
        """Invoke the sagemaker endpoint.

        Args:
            body (str): The body of the request.

        Returns:
            dict: The response from the endpoint.
        """
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=body,
            ContentType="application/json",
            Accept="application/json"
        )

        return json.loads(response['Body'].read().decode('utf-8'))


class HelsinkiItc(TranslateModel):
    """Helsinki NLP model."""

    def __init__(self, endpoint_name: str = "helsinki-nlp-opus-mt-itc-itc"):
        """Class constructor.

        Args:
            endpoint_name (str): The name of the endpoint to use.
        """
        super().__init__(endpoint_name)
        self.lang2iso = {
            Language.spanish: "spa",
            Language.italian: "ita",
            Language.portugees: "por",
            Language.catalan: "cat",
            Language.galician: "glg"
        }

    def predict(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the Helsinki NLP model.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language.
            target_lang (str): The target language.

        Returns:
            str: The translated text.
        """
        target_iso = self.lang2iso[target_lang]
        body = json.dumps({
            "inputs": f">>{target_iso}<< {text}",
        })
        prediction = self._invoke_sagemaker_endpoint(body=body)

        return prediction[0]["translation_text"]


class FacebookLocalMbart(TranslateModel):

    """Facebook MBART model local."""

    def __init__(self, endpoint_name: str = "mbart50_m2m"):
        """Class constructor.

        Args:
            endpoint_name (str): The name of the endpoint to use.
        """
        self.model = EasyNMT(endpoint_name)
        super().__init__(endpoint_name)
        self.lang2iso = {
            Language.spanish: "es",
            Language.italian: "it",
            Language.portugees: "pt",
        }

    def predict(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the Facebook MBART model.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language.
            target_lang (str): The target language.

        Returns:
            str: The translated text.
        """
        try:
            source_iso = self.lang2iso[source_lang]
            target_iso = self.lang2iso[target_lang]
        except KeyError:
            return "Language not supported"
        return self.model.translate(text, source_lang=source_iso, target_lang=target_lang)


class FacebookMbart(TranslateModel):
    """Facebook MBART model."""

    def __init__(self, endpoint_name: str = "facebook-mbart-large-50-many-to-many-mmt"):
        super().__init__(endpoint_name)
        self.lang2iso = {
            Language.spanish: "es_XX",
            Language.italian: "it_IT",
            Language.portugees: "pt_XX",
        }

    def predict(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the Facebook MBART model.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language.
            target_lang (str): The target language.

        Returns:
            str: The translated text.
        """
        try:
            source_iso = self.lang2iso[source_lang]
            target_iso = self.lang2iso[target_lang]
        except KeyError:
            return "Language not supported"

        body = json.dumps({
            'inputs': [text],
            "parameters": {
                'src_lang': source_iso,
                'tgt_lang': target_iso
            }
        })

        prediction = self._invoke_sagemaker_endpoint(body=body)

        return prediction[0]["translation_text"]


class AwsTranslate(TranslateModel):
    """AWS Translate model."""

    def __init__(self, endpoint_name: str = "aws-translate"):
        super().__init__(endpoint_name)
        self.client = boto3.client("translate")
        self.lang2iso = {
            Language.spanish: "es",
            Language.italian: "it",
            Language.portugees: "pt-PT",
            Language.catalan: "ca",
        }

    def predict(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the AWS Translate model.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language.
            target_lang (str): The target language.

        Returns:
            str: The translated text.
        """
        try:
            source_iso = self.lang2iso[source_lang]
            target_iso = self.lang2iso[target_lang]
        except KeyError:
            return "Language not supported"

        prediction = self.client.translate_text(
            Text=text,
            SourceLanguageCode=source_iso,
            TargetLanguageCode=target_iso
        )["TranslatedText"]

        return prediction
