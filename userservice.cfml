<cfcomponent 
    displayname="UserService"
    output="false"
    hint="Handles user operations, authentication, and AI-linked enrichment via BrainAI/NeoMind."
    rest="true"
    restpath="users">

    <!--- Constructor --->
    <cffunction name="init" access="public" returntype="any" output="false">
        <cfset variables.datasource = "main_db"> <!-- your CF DSN -->
        <cfreturn this>
    </cffunction>


    <!--- GET: /users/{id} --->
    <cffunction 
        name="getUser" 
        access="remote" 
        returntype="any" 
        httpmethod="GET" 
        restpath="{id}"
        output="false">

        <cfargument name="id" type="numeric" required="true">

        <cftry>

            <!-- Query user -->
            <cfquery name="qUser" datasource="#variables.datasource#">
                SELECT id, full_name, email, created_at
                FROM users
                WHERE id = <cfqueryparam value="#arguments.id#" cfsqltype="cf_sql_integer">
            </cfquery>

            <cfif qUser.recordCount EQ 0>
                <cfreturn { status="error", message="User not found." }>
            </cfif>

            <!-- Example AI enrichment call (mock) -->
            <cfset aiSummary = "This user profile can be enhanced via BrainAI and NeoMind embeddings.">

            <cfset response = {
                status = "success",
                data = {
                    id = qUser.id,
                    full_name = qUser.full_name,
                    email = qUser.email,
                    created_at = qUser.created_at,
                    enriched_summary = aiSummary
                }
            }>

            <cfreturn response>

            <cfcatch>
                <cfreturn { status="error", message="Internal server error.", detail=cfcatch.message }>
            </cfcatch>
        </cftry>
    </cffunction>



    <!--- POST: /users --->
    <cffunction 
        name="createUser" 
        access="remote" 
        returntype="any" 
        httpmethod="POST" 
        output="false"
        restpath="">

        <cfargument name="full_name" required="true">
        <cfargument name="email" required="true">

        <cftry>

            <!-- Insert new user -->
            <cfquery datasource="#variables.datasource#">
                INSERT INTO users (full_name, email)
                VALUES (
                    <cfqueryparam value="#arguments.full_name#" cfsqltype="cf_sql_varchar">,
                    <cfqueryparam value="#arguments.email#" cfsqltype="cf_sql_varchar">
                )
            </cfquery>

            <cfreturn {
                status="success",
                message="User created successfully.",
                data = {
                    full_name = arguments.full_name,
                    email = arguments.email
                }
            }>

            <cfcatch>
                <cfreturn { status="error", message="Creation failed.", detail=cfcatch.message }>
            </cfcatch>
        </cftry>

    </cffunction>



    <!--- GET: /users (all users) --->
    <cffunction 
        name="listUsers" 
        access="remote" 
        returntype="any" 
        httpmethod="GET" 
        output="false">

        <cftry>

            <cfquery name="qList" datasource="#variables.datasource#">
                SELECT id, full_name, email, created_at FROM users
                ORDER BY id DESC
            </cfquery>

            <cfset users = []>

            <cfloop query="qList">
                <cfset arrayAppend(users, {
                    id = qList.id,
                    full_name = qList.full_name,
                    email = qList.email,
                    created_at = qList.created_at
                })>
            </cfloop>

            <cfreturn {
                status="success",
                count = qList.recordCount,
                data = users
            }>

            <cfcatch>
                <cfreturn { status="error", message="Listing failed.", detail=cfcatch.message }>
            </cfcatch>

        </cftry>

    </cffunction>

</cfcomponent>
