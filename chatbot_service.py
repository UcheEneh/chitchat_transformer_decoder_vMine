""" New websocket code for chatbot integration.

"""
import json
import asyncio
import websockets

import chatbots
import soccerbot


from configparser import ConfigParser

# --- global parameter ------------------------------------------------------------------------------------------------


class ChatBotService:
    """
    """
    def __init__(self, path_to_config="chatbot_service_config.ini"):
        self.config = ConfigParser()
        self.config.read(path_to_config)
        self.receiver = None
        self.sender = None
        self.host = self.config.get('ChatBotService', 'host')
        self.port = self.config.get('ChatBotService', 'port')
        self.chatbots = []
        self.users = []
        self.__connection = False

    def run(self):
        print("ChatBotService: starting ...")
        self._function_initializer()
        self.__connection = True
        asyncio.get_event_loop().run_until_complete(self.receiver)
        print("ChatBotService: active ...")
        asyncio.get_event_loop().run_forever()

    def cancel(self):
        print("ChatBotService: canceling ...")
        self.__connection = False
        asyncio.get_event_loop().stop()

    def add_bot(self, chatbot):
        # b = chatbot.__bases__
        # if not isinstance(chatbot.__bases__, ChatBot):
        #     raise ValueError("Not a child of ChatBot.")  # TODO: check, if that works.
        self.chatbots.append({
            'obj': chatbot,
            'active': True  # TODO: outdated?
        })

    def _function_initializer(self):
        """ Websocket service
        """
        async def websocket_io(ws, path):
            while self.__connection:
                received_data = await ws.recv()
                print("Got Data: " + str(received_data))
                return_data = self._process_data(received_data)
                print("Return Data: " + str(return_data))
                await ws.send(json.dumps(return_data))
        self.receiver = websockets.serve(websocket_io, self.host, self.port)

    def _add_or_update_user(self, userid, active=None):
        """ Updates the status of a user and add the user, if not new. """
        if active is not None:
            for user in self.users:
                if user['userid'] == userid:
                    user['active'] = active
                    return
        else:
            active = False
        self.users.append({
            'userid': userid,
            'active': active
        })
        return

    def _get_user(self, userid):
        """ Returns the user specified with userid. """
        for user in self.users:
            if user['userid'] == userid:
                return user
        return None

    def _receiver_handler_chatbots(self, userid, utterance, nlu_result):
        """

        """
        results = []
        for chatbot in self.chatbots:
            result = {
                'chatbotSource': str(chatbot['obj']),
                'chatbotPossibleQualities': "Excellent,VeryGood,Good,Acceptable,Inacceptable"
            }
            try:
                answer, score, quality = chatbot['obj'].generate_answer(utterance=utterance,
                                                                        userid=userid,
                                                                        nlu_result=nlu_result)
                result['chatbotInput'] = utterance
                result['chatbotOutput'] = answer
                result['chatbotScore'] = score
                result['chatbotQuality'] = quality
                result['chatbotEntities'] = []
            except Exception as e:
                print("Chatbot Exception:")
                print(e)
                print("Bot: {}".format(str(chatbot['obj'])))
            finally:
                results.append(result)
        return results

    def _process_data(self, data):
        """ Core function
        """
        return_message = {
            "action": "chatbot.send",
            "params": {},
        }
        # --- decode received json message ----------------------------------------------------------------------------
        try:
            data_dict = json.loads(data)
        except json.decoder.JSONDecodeError:
            return_message['params']['error'] = "Not a valid json format."
            return return_message

        # --- check for meta and copy ---------------------------------------------------------------------------------
        if "meta" not in data_dict:
            return_message['params']['error'] = "No 'meta' found."
            return return_message
        else:
            self._add_or_update_user(data_dict['meta']['userId'])  # adds user, if needed
            return_message['meta'] = data_dict['meta']

        # --- handle params dict --------------------------------------------------------------------------------------
        if "params" not in data_dict:
            return_message['error'] = "No 'params' found."
            return return_message
        elif "res" in data_dict['params']:
            # --- result ----------------------------------------------------------------------------------------------
            # This is currently not implemented.
            # With the implicit mode, the dialogue history need to be tracked at all times.
            # ---------------------------------------------------------------------------------------------------------
            return_message['params']['info'] = "Got message."
        elif "req" in data_dict['params']:
            # --- request ---------------------------------------------------------------------------------------------
            # This is a users utterance. If the users chatbot mode is active, an answer is computed.
            # The result will be send to the dialogue manager.
            # ---------------------------------------------------------------------------------------------------------
            return_message['action'] = "dm.send"  # this should go to the dialogue manager
            return_message['params']['method'] = "chatbotResult"
            if "audioId" in data_dict['params']:
                return_message['params']['audioId'] = data_dict['params']['audioId']
            else:
                return_message['params']['audioId'] = "no audioId"
            if self._get_user(data_dict['meta']['userId'])['active']:
                if "nlu" in data_dict['params']:
                    nlu_result = data_dict['params']['nlu']
                else:
                    nlu_result = None
                chatbots_result = self._receiver_handler_chatbots(userid=data_dict['meta']['userId'],
                                                                  utterance=data_dict['params']['req'],
                                                                  nlu_result=nlu_result)
            else:
                chatbots_result = [{
                    "chatbotSource": "moviebot",
                    "chatbotInput": data_dict['params']['req'],
                    "chatbotOutput": "I can talk about movies now.",
                    "chatbotScore": 0.8,
                    "chatbotQuality": "VeryGood",
                    "chatbotPossibleQualities": "Excellent,VeryGood,Good,Acceptable,Inacceptable",
                    "chatbotEntities": []
                }, {
                    "chatbotSource": "soccerbot",
                    "chatbotInput": data_dict['params']['req'],
                    "chatbotOutput": "I can talk about football now.",
                    "chatbotScore": 0.8,
                    "chatbotQuality": "VeryGood",
                    "chatbotPossibleQualities": "Excellent,VeryGood,Good,Acceptable,Inacceptable",
                    "chatbotEntities": []
                }]
                if len(chatbots_result) < 1:
                    return_message['error'] = "Internal chatbot error."
            return_message['params']['payload'] = {
                "chatbotResults": chatbots_result}

            if "chatbotUseSpecificSource" in data_dict['params']:
                return_message['params']['payload']['chatbotUseSpecificSource'] = data_dict['params']['chatbotUseSpecificSource']

        elif "method" in data_dict['params']:
            # --- method ----------------------------------------------------------------------------------------------
            # If the dialogue manager sends a chatbot-state update, it is handled here.
            # ---------------------------------------------------------------------------------------------------------
            if data_dict['params']['method'] == "updateChatbotState":
                if data_dict['params']['payload']['chatbotActive']:
                    self._add_or_update_user(data_dict['meta']['userId'], active=True)
                    for chatbot in self.chatbots:
                        chatbot['obj'].delete_histories(userids=[data_dict['meta']['userId']])
                    return_message['params']['info'] = "Chatbot activated for the given user."
                else:
                    self._add_or_update_user(data_dict['meta']['userId'], active=False)
                    for chatbot in self.chatbots:
                        chatbot['obj'].delete_histories(userids=[data_dict['meta']['userId']])
                    return_message['params']['info'] = "Chatbot deactivated for the given user."
            else:
                return_message['params']['warning'] = "Did nothing."
        else:
            return_message['params']['warning'] = "Did nothing."
            return return_message
        # TODO remove this
        return return_message

#########
# NOTES #
#########
#  {"action":"dm.send","params":{"method":"nluResult","payload":{"hypotheses":[{"processedUtterance":"movie independence day","score":0.7343806316771934,"groups":[{"intents":[{"name":"ood-movie_generic","score":0.7343806316771934}],"entities":[{"id":1,"type":"movie_title","literal":"independence day","score":1}],"score":0.7343806316771934}]}],"rewriteNluVersion":"0.10.4"},"audioId":"8656a8e8-5f12-45a6-b59a-a04e6747016d","context":{"emotion":"neutral"},"locationData":{},"startOfSpeech":1558354078222,"nluAudioId":"0fa46d1f-e391-4cb7-8cf5-301e01a5ff0a","conversationId":"5ce298a0e04bfa6dd279354d"},"meta":{"requestId":"629","messageId":"629","userId":"5b1a4e8b97270640cd861b65","conversationId":"5ce298a0e04bfa6dd279354d"}}
#  {"action":"chatbot.send","params":{"req":"Tell me a","nlu":{"method":"nluResult","payload":{"hypotheses":[{"processedUtterance":"tell me a","score":0.31622427190743635,"groups":[{"intents":[{"name":"entertainment_do_joke","score":0.31622427190743635}],"entities":[],"score":0.31622427190743635}]}],"rewriteNluVersion":"0.10.4"},"audioId":"4c991aca-8069-444a-b3c4-057f2fccce2a"},"audioId":"f7255ad3-b199-4dcb-9d71-ac7c36bbf0a2"},"meta":{"requestId":"814","userId":"5b1a4e8b97270640cd861b65","conversationId":"5ce29728e04bfa6dd2793509","audioId":"f7255ad3-b199-4dcb-9d71-ac7c36bbf0a2"}}
#


if __name__ == "__main__":
    chbs = ChatBotService()
    chbs.add_bot(soccerbot.SoccerBot())
    chbs.add_bot(chatbots.BasicChatbot(path_params="save/moviecorpus/moviecorpus_clf_pipes=3_posemb_norm_37_utts/params.pkl",
                                       path_weights="save/moviecorpus/moviecorpus_clf_pipes=3_posemb_norm_37_utts/best_params.jl"))
    chbs.run()
