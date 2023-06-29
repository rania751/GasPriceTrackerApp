import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import LoginPage from "./pages/LoginPage";
import NotFound from "./pages/NotFound";
import Layout from "./pages/Layout";
import RegisterPage from "./pages/RegisterPage";
import Account from "./pages/Account";
import Result from "./pages/Result";
import Resultm from "./pages/Resultm";
import Predh from "./pages/Predh";
import Predm from "./pages/predm";
import Admin from "./pages/Admin"


const Router = () => {
  return (
 
    <BrowserRouter>
      <Routes>
         <Route path="/" element={<Layout> <LandingPage /> </Layout>} />
        <Route path="/login" element={<Layout> <LoginPage /> </Layout>} />
        <Route path="/admin" element={<Layout> <Admin />   <br/> <br/></Layout>} />
        <Route path="/register" element={<Layout> <RegisterPage /> </Layout>} />
        <Route path="/account" element={<Layout><br/><Predh /> <br/> <br/><Predm/></Layout>} />
        <Route path="/account/resultm" element={<Layout>  <Resultm />  </Layout>} />
        <Route path="/account/result" element={<Layout><br/><Result />  </Layout>} />
        <Route path="*" element={<Layout> <NotFound /> </Layout>} />

      </Routes>
    </BrowserRouter>
  );
};
export default Router;