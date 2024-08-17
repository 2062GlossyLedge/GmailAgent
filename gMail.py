from langchain_google_community import GmailToolkit


class gMailReader:

    def gMailTools():

        toolkit = GmailToolkit()

        from langchain_google_community.gmail.utils import (
            build_resource_service,
            get_gmail_credentials,
        )

        # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
        # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
        credentials = get_gmail_credentials(
            token_file="token.json",
            scopes=[
                "https://www.googleapis.com/auth/gmail.send",
            ],
            client_secrets_file="credentials.json",
        )
        api_resource = build_resource_service(credentials=credentials)
        toolkit = GmailToolkit(api_resource=api_resource)

        tools = toolkit.get_tools()

        print(tools)
        return tools
