from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = 'vasanth'
    age: Optional[int] = None,
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, description="CGPA must be between 0 and 10", default=7)


new_student = {'age': '28', 'email': 'abc@gmail.com'}
student = Student(**new_student)
student_json = student.model_dump_json()

