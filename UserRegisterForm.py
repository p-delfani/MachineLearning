import React from "react";
import "./UserRegisterForm.css";

export default class UserRegisterForm extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      firstName: "",
      lastName: "",
      email: "",
      errors: {},
      submitted: false
    };

    this.handleSubmit = this.handleSubmit.bind(this);
  }

  // متد عمومی برای همه input ها
  handleChange = (event) => {
    this.setState({
      [event.target.name]: event.target.value
    });
  };

  // اعتبارسنجی ایمیل با regex
  validateEmail(email) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  }

  // بررسی کل فرم
  validateForm() {
    let errors = {};

    if (!this.state.firstName.trim()) {
      errors.firstName = "First name is required";
    }

    if (!this.state.lastName.trim()) {
      errors.lastName = "Last name is required";
    }

    if (!this.state.email.trim()) {
      errors.email = "Email is required";
    } else if (!this.validateEmail(this.state.email)) {
      errors.email = "Email format is invalid";
    }

    this.setState({ errors });

    return Object.keys(errors).length === 0;
  }

  handleSubmit(event) {
    event.preventDefault();

    if (this.validateForm()) {
      this.setState({
        submitted: true,
        firstName: "",
        lastName: "",
        email: "",
        errors: {}
      });
    }
  }

  render() {
    return (
      <div className="form-container">

        {this.state.submitted && (
          <div className="success-message">
            Registration Successful 🎉
          </div>
        )}

        <form onSubmit={this.handleSubmit}>

          <input
            type="text"
            name="firstName"
            placeholder="First Name"
            value={this.state.firstName}
            onChange={this.handleChange}
          />
          {this.state.errors.firstName && (
            <span className="error">{this.state.errors.firstName}</span>
          )}

          <input
            type="text"
            name="lastName"
            placeholder="Last Name"
            value={this.state.lastName}
            onChange={this.handleChange}
          />
          {this.state.errors.lastName && (
            <span className="error">{this.state.errors.lastName}</span>
          )}

          <input
            type="text"
            name="email"
            placeholder="Email"
            value={this.state.email}
            onChange={this.handleChange}
          />
          {this.state.errors.email && (
            <span className="error">{this.state.errors.email}</span>
          )}

          <button type="submit">Register</button>

        </form>
      </div>
    );
  }
}UserRegisterForm
