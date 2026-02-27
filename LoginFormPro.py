import React, { useState } from "react";
import "./LoginFormPro.css";

export default function LoginFormPro() {

  const [formData, setFormData] = useState({
    username: "",
    password: ""
  });

  const [errors, setErrors] = useState({});
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // تغییر مقدار input ها
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // اعتبارسنجی فرم
  const validate = () => {
    let newErrors = {};

    if (!formData.username.trim()) {
      newErrors.username = "Username is required";
    }

    if (!formData.password.trim()) {
      newErrors.password = "Password is required";
    } else if (formData.password.length < 6) {
      newErrors.password = "Password must be at least 6 characters";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // سابمیت فرم
  const handleSubmit = (e) => {
    e.preventDefault();

    if (validate()) {
      setIsLoggedIn(true);
      setFormData({ username: "", password: "" });
      setErrors({});
    }
  };

  return (
    <div className="login-container">

      {isLoggedIn && (
        <div className="success-box">
          Login Successful 🚀
        </div>
      )}

      <form onSubmit={handleSubmit}>

        <input
          type="text"
          name="username"
          placeholder="Username"
          value={formData.username}
          onChange={handleChange}
        />
        {errors.username && (
          <span className="error-text">{errors.username}</span>
        )}

        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
        />
        {errors.password && (
          <span className="error-text">{errors.password}</span>
        )}

        <button type="submit">Login</button>

      </form>
    </div>
  );
}
