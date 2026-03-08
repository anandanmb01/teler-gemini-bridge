from pydantic import BaseModel


class CallFlowRequest(BaseModel):
    call_id: str
    account_id: str
    from_number: str
    to_number: str


class CallRequest(BaseModel):
    from_number: str
    to_number: str
    system_prompt: str = ""
    initial_prompt: str = ""
