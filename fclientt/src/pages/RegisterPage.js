import React, { useState } from "react";
import httpClient from "../httpClient";

export const RegisterPage = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [username, setUsername] = useState("");
    const [error, setError] = useState("");


  const registerUser = async () => {

  
    try {
      const resp = await httpClient.post("//localhost:5000/register", {
        email,
        password,
        username
      });
 
      window.location.href = "/";
    } catch (error) {
      if (error.response.status === 409) {
        setError("Username or Email already exists,Try again");
  

      }
    }
  };

  return (
<div class="container text-center">
<div class="row">
<div class="col">
</div>
<div class="col">

{/* <br/> 
<br/> 
  */}
  <br/> 
<br/> 

</div> 

<div class="container text-center">
    <div class="row">



    <div class="card h-100 shadow-none p-3 d-flex flex-column align-items-center" id="divLow">

      <h1>Create an account</h1>
      <form class="d-flex flex-column align-items-center">
    <div class="mb-3" id="root">
      <label for="exampleInputEmail1" class="form-label">Email</label>
      <input
        type="email"
        value={email}
        class="form-control"
        onChange={(e) => setEmail(e.target.value)}
        style={{ width: '200px', fontSize: '14px' }}
      />    

    </div>
    <div id="emailHelp" class="form-text">We'll never share your email with anyone else.</div>
      <br/>

      <div class="mb-3" id="root">
      <label for="exampleInputEmail1" class="form-label">   UserName</label>
      <input
                type="text"
                value={username}
                class="form-control"
                onChange={(e) => setUsername(e.target.value)}
                style={{ width: '200px', fontSize: '14px' }}
              />

      </div>
      <div class="mb-3" id="root">
      <label for="exampleInputPassword1" class="form-label">Password</label>
      <input
        type="password"
        value={password}
        class="form-control"
        onChange={(e) => setPassword(e.target.value)}
        style={{ width: '200px', fontSize: '14px' }}
      />

        </div>
        {/* <div class="mb-3 form-check">
        <input type="checkbox" class="form-check-input" id="exampleCheck1"/>
        <label class="form-check-label" for="exampleCheck1">Check me out</label>
        </div> */}
        <button type="button" class="btn btn-primary mt-2" onClick={() => registerUser()}>Submit</button>
        </form> 
        </div> 
        </div> 

</div>


{error ? (
          <div>
            <div>
            <br/>
            </div>
         
            <div className="alert alert-danger" role="alert">
              <span className="alert-icon">
                <span className="visually-hidden">Error</span>
              </span>
              <p>{error}</p>
            </div>
            <div>
            <div>
            </div>
            </div>
               
          </div>
          ):
          (<div></div>)}

<div >

</div> 
</div>
</div>
  );
};

export default RegisterPage;