component {
  this.name = "NeomindAI";
  this.sessionManagement = true;
  this.datasource = "brain_db";

  this.datasources = {
    "brain_db" = {
      class: "org.postgresql.Driver",
      connectionString: "jdbc:postgresql://db:5432/brain_ai",
      username: "brain",
      password: "secret"
    }
  };

  function onRequestStart(targetPage) {
    application.startTime = now();
  }
}
