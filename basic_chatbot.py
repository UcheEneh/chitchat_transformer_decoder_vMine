from abc import ABC, abstractmethod


class ChatBot(ABC):
    """ Abstract ChatBot class.
    Use this class as base to implement a ChatBot.
    """
    def __init__(self):
        super().__init__()

    # TODO: hier evtl. ein paar const. var. für den *args parameter von generate_answer!

    @abstractmethod
    def generate_answer(self, utterance, userid, nlu_result=None):
        """ Generates an answer.

        Args:
            utterance   A string. The user utterance.
            userid      An integer. The userid.
            nlu_result  A dict. (Optional) The result of the natural language understanding unit.

        Returns:
             A String   The answer.
             A float    A score between 0.0 and 1.0.
             A String   A discrete score. One of: Excellent, VeryGood, Good, Acceptable, Inacceptable.
        """
        return "", 0.0, ""

    @abstractmethod
    def delete_histories(self, userids, delete_all=False):
        """ Deletes dialogue histories.

        Args:
            userids     A list of int. The userids from which the histories should be deleted.
            delete_all  A boolean. Default is False. If True, all histories are deleted.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """ Should return the general name of the bot, e.g.: MovieBot."""
        pass

    @staticmethod
    def _get_nlu_intent(nlu_result, intent):
        """ Returns the correct nlu result, if available. """
        for hypothesis in nlu_result["payload"]["hypotheses"]:
            if "groups" not in hypothesis:
                return None
            for result in hypothesis['groups']:
                if result["intents"][0]["name"] == intent:
                    return result
        return None
