import React, { useState } from "react";
import httpClient from "../httpClient";

const LoginPage = () => {
const [email, setEmail] = useState("");
const [password, setPassword] = useState("");
const [error, setError] = useState("");
// const [valid, setValid] = useState("");
const logInUser = async () => {
   
  
  console.log(email, password);

    try {
      const resp = await httpClient.post("//localhost:5000/login", {
        email,
        password,
      });
      // setValid("successfully logged in");
      if (resp.data.email=="admin@gmail.com"){
        window.location.href = "/Admin";

      }
      else{window.location.href = "/Account";}
      
      
    } catch (error) {
      if (error.response.status === 401) {
        // alert("Invalid credentials");
        setError("Invalid credentials");

      }
    }
  };


return(



<div class="container text-center">
<div class="row">
<div class="col">
</div>
<div class="col">

{/* <br/> 
<br/>   */}
<br/> 
<br/> 
<br/> 
<br/> 
</div> 
<div class="card h-100 shadow-none p-3 d-flex flex-column align-items-center" id="divLow">
  <h1>Log Into Your Account</h1>
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
      <br/><br/>
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
 
    <button type="button" class="btn btn-primary mt-2" onClick={() => logInUser()}>
      Submit
    </button>
  </form>
</div>


<div class="col"> 
</div> 
</div>

{error ? (

  <div>
     <div>
              <br/>
              <br/>
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
              <br/>
            </div>
            </div>
          </div>
          ):(<div>
          </div>
          )}
          {/* {valid ? (
          <div class="alert alert-success" role="alert">
  <p>{valid}</p>
</div> ):(<div>
          </div>    
          )} */}
<div >
<br/> 
<br/> 
<br/> 
<br/> 
</div> 





</div>
);
};


export default  LoginPage




