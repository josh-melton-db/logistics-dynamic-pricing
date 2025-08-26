CREATE OR REPLACE FUNCTION manufacturing.dynamic_pricing._genie_query()()(
                databricks_host STRING, 
                databricks_token STRING,
                space_id STRING,
                question STRING,
                contextual_history STRING
) RETURNS STRING
LANGUAGE PYTHON
COMMENT 'This is a agent that you can converse with to get answers to questions. Try to provide simple questions and provide history if you had prior conversations.'
AS
$$
    import json
    import os
    import time
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Optional
    
    import pandas as pd
    import requests
    
    
    @dataclass
    class GenieResult:
        space_id: str
        conversation_id: str
        question: str
        content: Optional[str]
        sql_query: Optional[str] = None
        sql_query_description: Optional[str] = None
        sql_query_result: Optional[pd.DataFrame] = None
        error: Optional[str] = None
    
        def to_json_results(self):
            result = {
                "space_id": self.space_id,
                "conversation_id": self.conversation_id,
                "question": self.question,
                "content": self.content,
                "sql_query": self.sql_query,
                "sql_query_description": self.sql_query_description,
                "sql_query_result": self.sql_query_result.to_dict(
                    orient="records") if self.sql_query_result is not None else None,
                "error": self.error,
            }
            jsonified_results = json.dumps(result)
            return f"Genie Results are: {jsonified_results}"
    
        def to_string_results(self):
            results_string = self.sql_query_result.to_dict(orient="records") if self.sql_query_result is not None else None
            return ("Genie Results are: \n"
                    f"Space ID: {self.space_id}\n"
                    f"Conversation ID: {self.conversation_id}\n"
                    f"Question That Was Asked: {self.question}\n"
                    f"Content: {self.content}\n"
                    f"SQL Query: {self.sql_query}\n"
                    f"SQL Query Description: {self.sql_query_description}\n"
                    f"SQL Query Result: {results_string}\n"
                    f"Error: {self.error}")
    
    class GenieClient:
    
        def __init__(self, *,
                     host: Optional[str] = None,
                     token: Optional[str] = None,
                     api_prefix: str = "/api/2.0/genie/spaces"):
            self.host = host or os.environ.get("DATABRICKS_HOST")
            self.token = token or os.environ.get("DATABRICKS_TOKEN")
            assert self.host is not None, "DATABRICKS_HOST is not set"
            assert self.token is not None, "DATABRICKS_TOKEN is not set"
            self._workspace_client = requests.Session()
            self._workspace_client.headers.update({"Authorization": f"Bearer {self.token}"})
            self._workspace_client.headers.update({"Content-Type": "application/json"})
            self.api_prefix = api_prefix
            self.max_retries = 300
            self.retry_delay = 1
            self.new_line = "\r\n"
    
        def _make_url(self, path):
            return f"{self.host.rstrip('/')}/{path.lstrip('/')}"
    
        def start(self, space_id: str, start_suffix: str = "") -> str:
            path = self._make_url(f"{self.api_prefix}/{space_id}/start-conversation")
            resp = self._workspace_client.post(
                url=path,
                headers={"Content-Type": "application/json"},
                json={"content": "starting conversation" if not start_suffix else f"starting conversation {start_suffix}"},
            )
            resp = resp.json()
            print(resp)
            return resp["conversation_id"]
    
        def ask(self, space_id: str, conversation_id: str, message: str) -> GenieResult:
            path = self._make_url(f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages")
            # TODO: cleanup into a separate state machine
            resp_raw = self._workspace_client.post(
                url=path,
                headers={"Content-Type": "application/json"},
                json={"content": message},
            )
            resp = resp_raw.json()
            message_id = resp.get("message_id", resp.get("id"))
            if message_id is None:
                print(resp, resp_raw.url, resp_raw.status_code, resp_raw.headers)
                return GenieResult(content=None, error="Failed to get message_id")
    
            attempt = 0
            query = None
            query_description = None
            content = None
    
            while attempt < self.max_retries:
                resp_raw = self._workspace_client.get(
                    self._make_url(f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}"),
                    headers={"Content-Type": "application/json"},
                )
                resp = resp_raw.json()
                status = resp["status"]
                if status == "COMPLETED":
                    try:
    
                        query = resp["attachments"][0]["query"]["query"]
                        query_description = resp["attachments"][0]["query"].get("description", None)
                        content = resp["attachments"][0].get("text", {}).get("content", None)
                    except Exception as e:
                        return GenieResult(
                            space_id=space_id,
                            conversation_id=conversation_id,
                            question=message,
                            content=resp["attachments"][0].get("text", {}).get("content", None)
                        )
                    break
    
                elif status == "EXECUTING_QUERY":
                    self._workspace_client.get(
                        self._make_url(
                            f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result"),
                        headers={"Content-Type": "application/json"},
                    )
                elif status in ["FAILED", "CANCELED"]:
                    return GenieResult(
                        space_id=space_id,
                        conversation_id=conversation_id,
                        question=message,
                        content=None,
                        error=f"Query failed with status {status}"
                    )
                elif status != "COMPLETED" and attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return GenieResult(
                        space_id=space_id,
                        conversation_id=conversation_id,
                        question=message,
                        content=None,
                        error=f"Query failed or still running after {self.max_retries * self.retry_delay} seconds"
                    )
                attempt += 1
            resp = self._workspace_client.get(
                self._make_url(
                    f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result"),
                headers={"Content-Type": "application/json"},
            )
            resp = resp.json()
            columns = resp["statement_response"]["manifest"]["schema"]["columns"]
            header = [str(col["name"]) for col in columns]
            rows = []
            output = resp["statement_response"]["result"]
            if not output:
                return GenieResult(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    question=message,
                    content=content,
                    sql_query=query,
                    sql_query_description=query_description,
                    sql_query_result=pd.DataFrame([], columns=header),
                )
            for item in resp["statement_response"]["result"]["data_typed_array"]:
                row = []
                for column, value in zip(columns, item["values"]):
                    type_name = column["type_name"]
                    str_value = value.get("str", None)
                    if str_value is None:
                        row.append(None)
                        continue
                    match type_name:
                        case "INT" | "LONG" | "SHORT" | "BYTE":
                            row.append(int(str_value))
                        case "FLOAT" | "DOUBLE" | "DECIMAL":
                            row.append(float(str_value))
                        case "BOOLEAN":
                            row.append(str_value.lower() == "true")
                        case "DATE":
                            row.append(datetime.strptime(str_value, "%Y-%m-%d").date())
                        case "TIMESTAMP":
                            row.append(datetime.strptime(str_value, "%Y-%m-%d %H:%M:%S"))
                        case "BINARY":
                            row.append(bytes(str_value, "utf-8"))
                        case _:
                            row.append(str_value)
                rows.append(row)
    
            query_result = pd.DataFrame(rows, columns=header)
            return GenieResult(
                space_id=space_id,
                conversation_id=conversation_id,
                question=message,
                content=content,
                sql_query=query,
                sql_query_description=query_description,
                sql_query_result=query_result,
            )
    
    
    assert databricks_host is not None, "host is not set"
    assert databricks_token is not None, "token is not set"
    assert space_id is not None, "space_id is not set"
    assert question is not None, "question is not set"
    assert contextual_history is not None, "contextual_history is not set"
    client = GenieClient(host=databricks_host, token=databricks_token)
    conversation_id = client.start(space_id)
    formatted_message = f"""Use the contextual history to answer the question. The history may or may not help you. Use it if you find it relevant.
    
    Contextual History: {contextual_history}
    
    Question to answer: {question}
    """
    
    result = client.ask(space_id, conversation_id, formatted_message)
    
    return result.to_string_results()

$$;